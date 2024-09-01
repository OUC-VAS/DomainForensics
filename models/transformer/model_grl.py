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
from models.components.mmd import MMDLoss


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


class SSRTNet(nn.Module):
    def __init__(self, base_net='vit_base_patch16_224', use_bottleneck=True, bottleneck_dim=1024, width=1024,
                 class_num=31, args=None, use_freq=False, cfg=None, usemmd=False):
        super(SSRTNet, self).__init__()

        self.base_network = vit_model[base_net](pretrained=True, args=args, VisionTransformerModule=VT)
        self.use_bottleneck = use_bottleneck
        self.use_freq = use_freq
        self.use_multiscale_fusion = False

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
            self.freq = FreqFusionXFormerModule(dim_out=1024, depth=cfg.MODEL.FREQ_DEPTH, freq_dim=cfg.MODEL.FREQ_CHANNEL)
            # discriminator
            if not usemmd:
                self.discriminator_layer = [nn.Linear(classifier_dim*2, width), nn.ReLU(), nn.Dropout(0.0),
                                            nn.Linear(width, 1)]
        else:
            self.classifier_layer = [nn.Linear(classifier_dim, width), nn.ReLU(), nn.Dropout(0.0),
                                     nn.Linear(width, class_num)]
            self.classifier = nn.Sequential(*self.classifier_layer)
            if not usemmd:
                self.discriminator_layer = [nn.Linear(classifier_dim, width), nn.ReLU(), nn.Dropout(0.0),
                                            nn.Linear(width, 1)]
        if not usemmd:
            self.discriminator = nn.Sequential(*self.discriminator_layer)

        if self.use_bottleneck:
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)

        for dep in range(2):
            if not usemmd:
                self.discriminator[dep * 3].weight.data.normal_(0, 0.01)
                self.discriminator[dep * 3].bias.data.fill_(0.0)
            self.classifier[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier[dep * 3].bias.data.fill_(0.0)

        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=True)

        if not usemmd:
            self.parameter_list = [
                {"params": self.base_network.parameters(), "lr": 0.1},
                {"params": self.classifier.parameters(), "lr": 1},
                {"params": self.discriminator.parameters(), "lr": 1}
            ]
        else:
            self.parameter_list = [
                {"params": self.base_network.parameters(), "lr": 0.1},
                {"params": self.classifier.parameters(), "lr": 1},
            ]
        self.use_mmd = usemmd
        if self.use_bottleneck:
            self.parameter_list.extend([{"params": self.bottleneck.parameters(), "lr": 1}])
        if self.use_freq:
            self.parameter_list.extend([{"params": self.freq.parameters(), "lr": 1}])

    def forward(self, inputs, inputs_ycbcr=None):
        features, multi_scale_feats = self.base_network.forward_features(inputs)
        if self.use_bottleneck:
            features = self.bottleneck(features)
        if self.use_freq and inputs_ycbcr is not None:
            features = self.freq(inputs_ycbcr, features)
        if self.use_multiscale_fusion:
            features = self.multiscale_fusion(multi_scale_feats, features)

        outputs = self.classifier(features)
        if not self.use_mmd:
            outputs_dc = self.discriminator(self.grl(features))
            return features, outputs, outputs_dc
        return features, outputs


class SSRTRestoreManager():
    def __init__(self, network, cfg=None):
        super(SSRTRestoreManager, self).__init__()
        self.net = network
        self.cfg = cfg
        self.net_snapshot = None

        # restore variables
        self.sr_loss_weight = 0.2 # cfg.SR_LOSS_WEIGHT
        self.iter_num = 0
        self.restore = False
        self.r = 0.0
        self.r_period = 1000  # args.adap_adjust_T
        self.r_phase = 0
        self.r_mag = 1.0
        self.adap_adjust_T = 1000  # args.adap_adjust_T
        self.adap_adjust_L = 4  # args.adap_adjust_L
        self.adap_adjust_append_last_subintervals = True  # args.adap_adjust_append_last_subintervals
        self.adap_adjust_last_restore_iter = 0
        self.divs = []
        self.divs_last_period = None

    def check_restore_status(self, out1):

        if self.restore and self.iter_num > 0 and self.sr_loss_weight > 0:
            self.restore_snapshot()
            self.restore = False

        if self.iter_num % self.adap_adjust_T == 0 and self.sr_loss_weight > 0:
            self.save_snapshot()
            self.divs = []

        prob1 = F.softmax(out1, dim=1)
        self.r = self.get_adjust(self.iter_num)
        self.net.base_network.sr_alpha_adap = self.net.base_network.sr_alpha * self.r
        self.sr_loss_weight_adap = self.sr_loss_weight * self.r

        div_unique = prob1.argmax(-1).unique().shape[0]
        self.divs.append(div_unique)
        if (self.iter_num + 1) % self.adap_adjust_T == 0 and self.iter_num > 0:
            self.check_div_drop()
            if not self.restore:
                self.divs_last_period = self.divs

        self.iter_num += 1

    def save_snapshot(self):
        self.net_snapshot = self.net.state_dict()
        print("Safe Training : Save Net and Optim snapshot ... ")

    def restore_snapshot(self):
        self.net.load_state_dict(self.net_snapshot)
        self.adap_adjust_last_restore_iter = self.iter_num
        print("Safe Training : Restore Net and Optim snapshot ... ")

    def check_div_drop(self):
        flag = False

        for l in range(self.adap_adjust_L+1):
            chunk = np.power(2, l)
            divs_ = np.array_split(np.array(self.divs), chunk)
            divs_ = [d.mean() for d in divs_]

            if self.adap_adjust_append_last_subintervals and self.divs_last_period is not None:
                divs_last_period = np.array_split(np.array(self.divs_last_period), chunk)
                divs_last_period = [d.mean() for d in divs_last_period]
                divs_.insert(0, divs_last_period[-1])

            for i in range(len(divs_)-1):
                if divs_[i+1] < divs_[i] - 1.0:
                    flag = True

        if self.r <= 0.1:
            flag = False

        if flag:
            self.restore = True
            self.r_phase = self.iter_num
            if self.iter_num - self.adap_adjust_last_restore_iter <= self.r_period:
                self.r_period *= 2

    def get_adjust(self, iter):
        if iter >= self.r_period + self.r_phase:
            return self.r_mag
        return np.sin((iter - self.r_phase) / self.r_period * np.pi / 2) * self.r_mag


class SSRT(object):
    def __init__(self, base_net='vit_base_patch16_224', bottleneck_dim=1024, class_num=31, use_gpu=True, source_only=False, args=None, cfg=None):
        use_bottleneck = True
        self.cfg = cfg
        use_freq = self.cfg.DATAS.WITH_FREQUENCY
        use_mmd = True

        self.net = SSRTNet(base_net, use_bottleneck, bottleneck_dim, bottleneck_dim,
                           class_num, args, use_freq=use_freq, cfg=cfg, usemmd=use_mmd)
        self.temperature = 0.5
        self.base_temperature = 0.07
        self.local_alignment = self.cfg.MODEL.LOCAL_ALIGNMENT
        self.source_only = source_only
        self.warm_epoch = -1
        self.mmd_loss = MMDLoss() if use_mmd else None

    def get_loss(self, inputs_source, inputs_target=None, labels_source=None, labels_target=None, inputs_s_ycbcr=None, inputs_t_ycbcr=None,
                 input_source_augs=None, queue_target=None, queue_source=None, discriminator=None, cur_epoch=None):
        if cur_epoch > self.warm_epoch:
            domain_weight = 1.0
        else:
            domain_weight = 0.0
        # Adaptation loss mode
        inputs = torch.cat((inputs_source, inputs_target))
        if inputs_s_ycbcr is not None:
            inputs_ycbcr = torch.cat((inputs_s_ycbcr, inputs_t_ycbcr))
            if self.mmd_loss is not None:
                features, outputs = self.net(inputs, inputs_ycbcr)
            else:
                features, outputs, outputs_dc = self.net(inputs, inputs_ycbcr)
        else:
            if self.mmd_loss is not None:
                features, outputs, outputs_dc = self.net(inputs)
            else:
                features, outputs = self.net(inputs)

        classification_loss = nn.CrossEntropyLoss()(outputs.narrow(0, 0, labels_source.size(0)), labels_source)

        if self.mmd_loss is not None:
            domain_loss = self.mmd_loss(outputs.narrow(0, 0, labels_source.size(0)), outputs.narrow(0, labels_source.size(0), labels_source.size(0)))
        else:
            domain_labels = torch.cat(
                (torch.ones(inputs_source.shape[0], device=inputs.device, dtype=torch.float),
                 torch.zeros(inputs_target.shape[0], device=inputs.device, dtype=torch.float)),
                0)
            domain_loss = nn.BCELoss()(F.sigmoid(outputs_dc.narrow(0, 0, inputs.size(0))).squeeze(), domain_labels) * 2

        total_loss = classification_loss + domain_weight * domain_loss
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


if __name__ == '__main__':
    from configs.defaults import get_config
    cfg = get_config()
    model = SSRT(class_num=2, cfg=cfg)
    del model.net.base_network.head

    model.net
    data = torch.randn(2,3,224,224)
    _, out, out_dc = model.net(data)
    print(out[0].shape)
    print(model.net)
    print('done')
