import torch.nn as nn

from torchvision import models
import torch.nn.functional as F
from models.components.freq_mod import FreqFusionModule


class ResnetBase(nn.Module):
    def __init__(self, model='resnet50', pretrained=False):
        super(ResnetBase, self).__init__()
        if model == 'resnet50':
            model_ft = models.resnet50(pretrained=pretrained)
        elif model == 'resnet101':
            model_ft = models.resnet101(pretrained=pretrained)
        else:
            model_ft = models.resnet50(pretrained=pretrained)
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)
        self.dim=2048

    def forward(self, x):
        out = self.features(x)
        out = out.view(x.size(0), self.dim)
        return out


class ResClassifierMME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05):
        super(ResClassifierMME, self).__init__()
        self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.tmp = temp
        self.norm = True

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
        else:
            x = self.fc(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def weights_init(self, m):
        m.weight.data.normal_(0.0, 0.1)


class ResNetWithC(nn.Module):
    def __init__(self, class_num=2, cfg=None):
        super(ResNetWithC, self).__init__()
        self.cfg = cfg
        model_name = cfg.MODEL.NAME
        self.net = ResnetBase(model=model_name, pretrained=True)
        self.use_freq = self.cfg.DATAS.WITH_FREQUENCY

        classifier_dim = 2048
        width = classifier_dim
        if self.use_freq:
            self.classifier_layer = [nn.Linear(classifier_dim + 1024, width), nn.ReLU(),
                                     nn.Linear(width, class_num)]
            self.classifier = nn.Sequential(*self.classifier_layer)
            self.freq = FreqFusionModule(dim_out=classifier_dim)
        else:
            self.classifier_layer = [nn.Linear(classifier_dim, width), nn.ReLU(),
                                     nn.Linear(width, class_num)]
            self.classifier = nn.Sequential(*self.classifier_layer)


        self.parameter_list = [
            {"params": self.net.parameters(), "lr": 1},
            {"params": self.classifier.parameters(), "lr": 1}
        ]

        if self.use_freq:
            self.parameter_list.extend([{"params": self.freq.parameters(), "lr": 1}])

    def forward(self, x, x_ycbcr=None):
        features = self.net(x)
        if self.use_freq and x_ycbcr is not None:
            features = self.freq(x_ycbcr, features)
        outputs = self.classifier(features)
        return features, outputs


if __name__ == '__main__':
    import torch
    from configs.defaults import get_config
    cfg = get_config()
    cfg.DATAS.WITH_FREQUENCY = False
    model = ResNetWithC(class_num=2, cfg=cfg)

    data = torch.randn(2,3,224,224)
    freq_data = torch.randn(2, 192, 28, 28)

    _, out = model(data) # , freq_data
    # dist_loss = model.self_distillation(out, out)
    print(_.shape)
    print(model.net)
    print('done')
