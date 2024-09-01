# ref: https://github.com/tsun/SSRT

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import random
import numpy as np

from models.transformer.vit import Block, PatchEmbed, VisionTransformer, vit_model
from models.components.grl import WarmStartGradientReverseLayer
from models.components.freq_mod import FreqFusionModule, MultiScaleFusion, FreqFusionXFormerModule


class VT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., distilled=False,
                 args=None, cfg=None):

        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.distilled = distilled

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        self.output_block = [0, 3, 7, 11]
        # self.output_block = cfg.MODEL.VIT_OUT_BLOCKS

    def forward_features(self, x):
        B = x.shape[0]
        feat_outputs = []
        y = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(y.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        if self.distilled:
            dist_tokens = self.dist_token.expand(y.shape[0], -1, -1)
            y = torch.cat((cls_tokens, dist_tokens, y), dim=1)
        else:
            y = torch.cat((cls_tokens, y), dim=1)
        y = y + self.pos_embed
        y = self.pos_drop(y)

        for layer, blk in enumerate(self.blocks):
            y = blk(y)
            if layer in self.output_block:
                feat_outputs.append(y)

        y = self.norm(y)
        y = y[:, 0]

        return y, feat_outputs


class ViTBaseNet(nn.Module):
    def __init__(self, base_net='vit_base_patch16_224', use_bottleneck=True, bottleneck_dim=1024, width=1024,
                 class_num=31, args=None, use_freq=False, cfg=None):
        super(ViTBaseNet, self).__init__()

        self.base_network = vit_model[base_net](pretrained=True, args=args, VisionTransformerModule=VT)
        self.use_bottleneck = use_bottleneck
        self.use_freq = use_freq

        if self.use_bottleneck:
            self.bottleneck_layer = [nn.Linear(self.base_network.embed_dim, bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.0)]
            self.bottleneck = nn.Sequential(*self.bottleneck_layer)

        classifier_dim = bottleneck_dim if use_bottleneck else self.base_network.embed_dim
        # self.classifier_layer = [nn.Linear(classifier_dim * 2, width), nn.ReLU(), nn.Dropout(0.5), nn.Linear(width, class_num)]
        # self.classifier = nn.Sequential(*self.classifier_layer)

        if self.use_freq:
            self.classifier_layer = [nn.Linear(classifier_dim * 2, width), nn.ReLU(), nn.Dropout(0.0),
                                     nn.Linear(width, class_num)]
            self.classifier = nn.Sequential(*self.classifier_layer)
            # self.freq = FreqFusionModule(dim_out=1024)
            self.freq = FreqFusionXFormerModule(dim_out=1024, depth=cfg.MODEL.FREQ_DEPTH, freq_dim=cfg.MODEL.FREQ_CHANNEL) # cfg.MODEL.FREQ_DEPTH # cfg.MODEL.FREQ_CHANNEL
        else:
            self.classifier_layer = [nn.Linear(classifier_dim, width), nn.ReLU(), nn.Dropout(0.0),
                                     nn.Linear(width, class_num)]
            self.classifier = nn.Sequential(*self.classifier_layer)

        if self.use_bottleneck:
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)

        for dep in range(2):
            self.classifier[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier[dep * 3].bias.data.fill_(0.0)

        self.parameter_list = [
            {"params": self.base_network.parameters(), "lr": 0.1},
            {"params": self.classifier.parameters(), "lr": 1}
        ]
        if self.use_bottleneck:
            self.parameter_list.extend([{"params": self.bottleneck.parameters(), "lr": 1}])
        if self.use_freq:
            self.parameter_list.extend([{"params": self.freq.parameters(), "lr": 1}])

    def forward(self, inputs, inputs_ycbcr=None):
        features, _ = self.base_network.forward_features(inputs)
        if self.use_bottleneck:
            features = self.bottleneck(features)
        if self.use_freq and inputs_ycbcr is not None:
            features = self.freq(inputs_ycbcr, features)
        outputs = self.classifier(features)
        return features, outputs


class BANetWrapper(object):
    def __init__(self, base_net='vit_base_patch16_224', bottleneck_dim=1024, class_num=31, use_gpu=True, source_only=False, args=None, cfg=None):
        use_bottleneck = True
        self.cfg = cfg
        use_freq = self.cfg.DATAS.WITH_FREQUENCY
        self.net = ViTBaseNet(base_net, use_bottleneck, bottleneck_dim, bottleneck_dim,
                           class_num, args, use_freq=use_freq, cfg=cfg)
        self.temperature = 0.5
        self.local_alignment = self.cfg.MODEL.LOCAL_ALIGNMENT
        self.source_only = source_only
        self.warm_epoch = -1
        self.use_proto_align = False
        self.ent_min = False

    def get_disc_loss(self, feature, label):
        disc_loss = nn.BCEWithLogitsLoss()(feature, label)
        return disc_loss

    def discriminator_loss(self, s_x, t_x, discriminator, s_ycbcr=None, t_ycbcr=None, cur_epoch=None):
        if cur_epoch is not None:
            if cur_epoch > self.warm_epoch:
                domain_weight = 1.0
            else:
                domain_weight = 0.0
        else:
            domain_weight = 1.0
        inputs = torch.cat((s_x, t_x))
        with torch.no_grad():
            if s_ycbcr is not None:
                inputs_ycbcr = torch.cat((s_ycbcr, t_ycbcr))
                features, _ = self.net(inputs, inputs_ycbcr)
            else:
                features, _ = self.net(inputs)
        features = discriminator(features.detach())
        real_loss = self.get_disc_loss(features.narrow(0, 0, features.size()[0]//2).squeeze(), torch.zeros(features.size()[0]//2).to(features.device))
        fake_loss = self.get_disc_loss(features.narrow(0, features.size()[0] // 2, features.size()[0] // 2).squeeze(),
                                       torch.ones(features.size()[0] // 2).to(features.device))
        discriminator_loss = real_loss + fake_loss
        return discriminator_loss * domain_weight

    def self_distillation(self, logits, labels):
        logits = torch.exp(logits / self.temperature) / torch.sum(torch.exp(logits / self.temperature), dim=1, keepdim=True)
        labels = torch.exp(labels / self.temperature) / torch.sum(torch.exp(labels / self.temperature), dim=1, keepdim=True)

        distill_loss = -torch.sum(labels * torch.log(logits + 1e-4), dim=1)

        distill_loss = distill_loss.mean()
        return distill_loss

    def get_finetune_loss(self, s_x, s_y, t_x, t_y, s_ycbcr=None, t_ycbcr=None, discriminator=None, lbl_target=None, return_ce=False):
        inputs = torch.cat((s_x, t_x))
        if s_ycbcr is not None:
            inputs_ycbcr = torch.cat((s_ycbcr, t_ycbcr), dim=0)
            features, outputs = self.net(inputs, inputs_ycbcr)
        else:
            features, outputs = self.net(inputs)

        if self.ent_min:
            distill_loss = self.entropy_mininization(logits=outputs.narrow(0, s_y.size()[0], s_y.size()[0]))
        else:
            distill_loss = self.self_distillation(logits=outputs.narrow(0, s_y.size()[0], t_y.size()[0]), labels=t_y)
        if return_ce:
            target_ce_loss = nn.CrossEntropyLoss()(outputs.narrow(0, s_y.size(0), t_y.size(0)),
                                                   lbl_target)

        # finetune self-distillation loss
        if discriminator is not None:
            fake_d = discriminator(features.narrow(0, 0, s_y.size(0)))
            domain_loss = self.get_disc_loss(fake_d.squeeze(), torch.ones(fake_d.size()[0]).to(fake_d.device))
            # finetune disc loss
            total_loss = distill_loss + domain_loss
            if self.use_proto_align:
                total_loss = total_loss
                return total_loss, distill_loss.detach(), domain_loss.detach()
            else:
                if return_ce:
                    return total_loss, distill_loss.detach(), domain_loss.detach(), target_ce_loss.detach()
                else:
                    return total_loss, distill_loss.detach(), domain_loss.detach()
        else:
            total_loss = distill_loss
            if self.use_proto_align:
                total_loss = total_loss
                return total_loss, distill_loss.detach()
            else:
                return total_loss, distill_loss.detach()

    def get_loss(self, inputs_source, inputs_target=None, labels_source=None, labels_target=None, inputs_s_ycbcr=None, inputs_t_ycbcr=None,
                 input_source_augs=None, queue_target=None, queue_source=None, discriminator=None, cur_epoch=None, reture_ce=False):

        # Source only loss mode

        if self.source_only:
            if inputs_s_ycbcr is not None:
                features, outputs = self.net(inputs_source, inputs_s_ycbcr)
            else:
                features, outputs = self.net(inputs_source)
            classification_loss = nn.CrossEntropyLoss()(outputs.narrow(0, 0, labels_source.size(0)), labels_source)
            if reture_ce:
                with torch.no_grad():
                    target_ce_loss = nn.CrossEntropyLoss()(outputs.narrow(0, labels_source.size(0), labels_source.size(0)), labels_target)

            if self.local_alignment:
                if queue_target is not None:
                    if torch.any(queue_target.sum(1) == 0):
                        local_alignment_loss = 0.0 * features.mean()
                    else:
                        local_alignment_loss = self.local_alignment_loss(features.narrow(0, 0, labels_source.size(0)),
                                                                         queue_target)
                else:
                    local_alignment_loss = 0.0 * features.mean()
                total_loss = classification_loss + 0.1 * local_alignment_loss
                return total_loss, classification_loss.detach(), local_alignment_loss.detach(), features.narrow(0, 0, labels_source.size(0))
            else:
                if reture_ce:
                    return classification_loss, target_ce_loss.detach()
                else:
                    return classification_loss

        if cur_epoch > self.warm_epoch:
            domain_weight = 1.0
        else:
            domain_weight = 0.0
        # Adaptation loss mode
        inputs = torch.cat((inputs_source, inputs_target))
        if inputs_s_ycbcr is not None:
            inputs_ycbcr = torch.cat((inputs_s_ycbcr, inputs_t_ycbcr))
            features, outputs = self.net(inputs, inputs_ycbcr)
        else:
            features, outputs = self.net(inputs)

        classification_loss = nn.CrossEntropyLoss()(outputs.narrow(0, 0, labels_source.size(0)), labels_source)
        if reture_ce:
            with torch.no_grad():
                target_ce_loss = nn.CrossEntropyLoss()(outputs.narrow(0, labels_source.size(0), labels_target.size(0)),
                                                       labels_target)

        fake_d = discriminator(features.narrow(0, labels_source.size(0), labels_source.size(0)))
        domain_loss = self.get_disc_loss(fake_d.squeeze(), torch.zeros(fake_d.size()[0]).to(fake_d.device))

        if self.local_alignment:
            if queue_target is not None:
                if torch.any(queue_target.sum(1) == 0):
                    local_alignment_loss = 0.0 * features.mean()
                else:
                    local_alignment_loss = self.local_alignment_loss(features.narrow(0, labels_source.size(0), labels_source.size(0)), queue_target)
            else:
                local_alignment_loss = 0.0 * features.mean()
            total_loss = classification_loss + domain_weight * domain_loss + 0.5 * local_alignment_loss
            return total_loss, classification_loss.detach(), domain_loss.detach(), local_alignment_loss.detach(), features.narrow(0, 0, labels_source.size(0))
        else:
            total_loss = classification_loss + domain_weight * domain_loss
            if reture_ce:
                return total_loss, classification_loss.detach(), domain_loss.detach(), target_ce_loss.detach()
            else:
                return total_loss, classification_loss.detach(), domain_loss.detach()

    def predict(self, inputs, output='prob'):
        outputs = self.net(inputs)
        if output == 'prob':
            softmax_outputs = F.softmax(outputs)
            return softmax_outputs
        elif output == 'score':
            return outputs
        else:
            raise NotImplementedError('Invalid output')

    def get_parameter_list(self):
        return self.net.parameter_list
