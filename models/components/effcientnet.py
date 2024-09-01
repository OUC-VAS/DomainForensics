from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from models.components.freq_mod import FreqFusionModule


class EfficientNetBackbone(EfficientNet):

    def forward(self, inputs):
        # Convolution layers
        x = self.extract_features(inputs)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
        return x


def get_efficientnet(name='efficientnet-b0'):
    model = EfficientNetBackbone.from_pretrained(model_name=name)
    del model._fc
    return model


class EfficientNetWithC(nn.Module):
    def __init__(self, class_num=2, model_name='efficientnet-b0', cfg=None):
        super(EfficientNetWithC, self).__init__()
        self.cfg = cfg
        self.net = get_efficientnet(model_name)

        self.use_freq = self.cfg.DATAS.WITH_FREQUENCY

        if model_name == 'efficientnet-b0':
            classifier_dim = 1280
        elif model_name == 'efficientnet-b4':
            classifier_dim = 1792
        else:
            classifier_dim = 1280
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

    # model = get_efficientnet('efficientnet-b4')
    model = EfficientNetWithC(class_num=2, cfg=cfg)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y[0].size())
    print(y[1].size())
    print(model)
    print('done')
