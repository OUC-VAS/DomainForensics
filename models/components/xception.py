"""
[refs:] (https://github.com/FanDady/Face-Forgery-Detection/blob/master/code/networks/xception.py)

Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from models.components.freq_mod import FreqFusionModule

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        # x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model


def return_pytorch04_xception(pretrained=True):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained=False)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear

        settings = pretrained_settings['xception']['imagenet']
        state_dict = model_zoo.load_url(settings['url'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(
                    -1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        del model.fc
    return model


class XceptionNet(nn.Module):
    def __init__(self, class_num=2, cfg=None):
        super(XceptionNet, self).__init__()
        self.cfg = cfg
        self.net = return_pytorch04_xception(pretrained=True)
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

        # for dep in range(2):
        #     self.classifier[dep * 3].weight.data.normal_(0, 0.01)
        #     self.classifier[dep * 3].bias.data.fill_(0.0)

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


class XceptionModel(object):
    def __init__(self, cfg=None, source_only=False):
        super(XceptionModel, self).__init__()
        self.net = XceptionNet(class_num=2, cfg=cfg)
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

if __name__ == '__main__':

    from configs.defaults import get_config
    cfg = get_config()
    cfg.DATAS.WITH_FREQUENCY = False
    model = XceptionModel(cfg=cfg)

    data = torch.randn(2,3,299,299)
    freq_data = torch.randn(2, 192, 28, 28)

    _, out = model.net(data) # , freq_data
    dist_loss = model.self_distillation(out, out)
    print(out[0].shape)
    print(model.net)
    print('done')
