import torch
import torch.nn as nn

from models.components.xception import XceptionNet
from models.components.effcientnet import EfficientNetWithC
from models.components.resnet import ResNetWithC

def get_model(cfg=None):
    if cfg.MODEL.NAME == 'xception':
        model = XceptionNet(class_num=2, cfg=cfg)
        class_feat_dim = 2048
    elif cfg.MODEL.NAME == 'efficientnet-b0':
        model = EfficientNetWithC(class_num=2, model_name=cfg.MODEL.NAME, cfg=cfg)
        class_feat_dim = 1280
    elif cfg.MODEL.NAME == 'efficientnet-b4':
        model = EfficientNetWithC(class_num=2, model_name=cfg.MODEL.NAME, cfg=cfg)
        class_feat_dim = 1792
    elif cfg.MODEL.NAME == 'resnet50':
        model = ResNetWithC(class_num=2, cfg=cfg)
        class_feat_dim = 2048
    elif cfg.MODEL.NAME == 'resnet101':
        model = ResNetWithC(class_num=2, cfg=cfg)
        class_feat_dim = 2048
    else:
        model = XceptionNet(class_num=2, cfg=cfg)
        class_feat_dim = 2048
    return model, class_feat_dim


class TransferModel(object):
    def __init__(self, cfg=None, source_only=False):
        super(TransferModel, self).__init__()
        self.net, self.classifier_dim = get_model(cfg)
        self.source_only = source_only
        self.warm_epoch = -1
        self.temperature = 0.5

    def get_disc_loss(self, feature, label):
        disc_loss = nn.BCEWithLogitsLoss()(feature, label)
        return disc_loss

    def get_loss(self, inputs_source, inputs_target=None, labels_source=None, labels_target=None, inputs_s_ycbcr=None, inputs_t_ycbcr=None,
                 input_source_augs=None, queue_target=None, queue_source=None, discriminator=None, cur_epoch=None):
        if self.source_only:
            if inputs_s_ycbcr is not None:
                features, outputs = self.net(inputs_source, inputs_s_ycbcr)
            else:
                features, outputs = self.net(inputs_source)
            classification_loss = nn.CrossEntropyLoss()(outputs.narrow(0, 0, labels_source.size(0)), labels_source)
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

        fake_d = discriminator(features.narrow(0, labels_source.size(0), labels_source.size(0)))
        domain_loss = self.get_disc_loss(fake_d.squeeze(), torch.zeros(fake_d.size()[0]).to(fake_d.device))

        total_loss = classification_loss + domain_weight * domain_loss
        return total_loss, classification_loss.detach(), domain_loss.detach()

    def self_distillation(self, logits, labels):
        logits = torch.exp(logits / self.temperature) / torch.sum(torch.exp(logits / self.temperature), dim=1, keepdim=True)
        labels = torch.exp(labels / self.temperature) / torch.sum(torch.exp(labels / self.temperature), dim=1, keepdim=True)

        distill_loss = -torch.sum(labels * torch.log(logits + 1e-4), dim=1)

        distill_loss = distill_loss.mean()
        return distill_loss

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

    def get_finetune_loss(self, s_x, s_y, t_x, t_y, s_ycbcr=None, t_ycbcr=None, discriminator=None):
        inputs = torch.cat((s_x, t_x))
        if s_ycbcr is not None:
            inputs_ycbcr = torch.cat((s_ycbcr, t_ycbcr), dim=0)
            features, outputs = self.net(inputs, inputs_ycbcr)
        else:
            features, outputs = self.net(inputs)

        distill_loss = self.self_distillation(logits=outputs.narrow(0, s_y.size()[0], t_x.size()[0]), labels=t_y)

        # finetune self-distillation loss
        if discriminator is not None:
            fake_d = discriminator(features.narrow(0, 0, s_y.size(0)))
            domain_loss = self.get_disc_loss(fake_d.squeeze(), torch.ones(fake_d.size()[0]).to(fake_d.device))
            # finetune disc loss
            total_loss = distill_loss + domain_loss

            return total_loss, distill_loss.detach(), domain_loss.detach()
        else:
            total_loss = distill_loss
            return total_loss, distill_loss.detach()

    def get_parameter_list(self):
        return self.net.parameter_list
