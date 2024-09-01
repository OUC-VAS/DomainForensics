import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.transformer.vit import Block
from functools import partial


class FreqFusionXFormerModule(nn.Module):
    def __init__(self, dim_out=1024, depth=4, freq_dim=768):
        super(FreqFusionXFormerModule, self).__init__()
        # conv
        dim_in = 3*64
        dim_mid = freq_dim
        embed_dim = dim_mid
        div_ratio = 4
        depth = depth
        num_heads = 12
        mlp_ratio = 4
        qkv_bias = True,
        drop_rate = 0.1
        attn_drop_rate = 0.
        drop_path_rate = 0.
        distilled = False
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = 224 //16 * 224//16

        self.stem_convs = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), groups=3),
            nn.BatchNorm2d(dim_in),
            # norm_layer(dim_in),
            nn.GELU(),
            # nn.ReLU(), # n, 192, 14, 14
            nn.Conv2d(dim_in, dim_mid, kernel_size=(1, 1), stride=(1, 1), padding=(0,0)),
            nn.BatchNorm2d(dim_mid),
            # norm_layer(dim_mid),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_mid, kernel_size=(1, 1), stride=(1, 1), padding=(0,0)),
            nn.BatchNorm2d(dim_mid),
            # norm_layer(dim_mid),
            nn.GELU(),  # n, 768, 14 , 14
        )
        # change to layer norm
        self.blks = nn.Sequential(
            *[Block(dim=dim_mid, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer) for i in range(depth)]
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = norm_layer(embed_dim)

        self.bottleneck = nn.Sequential(
            nn.Linear(dim_mid, dim_out),
            norm_layer(dim_out),
            nn.GELU()
        )
        self.bottleneck[0].weight.data.normal_(0, 0.005)
        self.bottleneck[0].bias.data.fill_(0.1)

    def forward(self, x_ycbcr, rgb_feat):
        """
        :param x: ycbcr -> dct -> x
        :return: frequency features
        """
        freq_feat = self.stem_convs(x_ycbcr)  # N, 768, 14, 14
        b, c, _, _ = freq_feat.size()
        freq_feat = freq_feat.view(b, c, -1).permute(0, 2, 1).contiguous()

        cls_tokens = self.cls_token.expand(freq_feat.shape[0], -1, -1)
        freq_feat = torch.cat((cls_tokens, freq_feat), dim=1)
        freq_feat = freq_feat + self.pos_embed
        for layer, blk in enumerate(self.blks):
            freq_feat = blk(freq_feat)
        freq_feat = self.norm(freq_feat)[:, 0]
        freq_feat = self.bottleneck(freq_feat)

        # freq_attn = self.channel_attn(freq_feat)
        # freq_feat = freq_feat + freq_feat * freq_attn
        # freq_feat = self.freq_pooling(freq_feat).flatten(start_dim=1)
        all_feat = torch.cat((rgb_feat, freq_feat), dim=1)
        # all_feat = self.fusion_conv(all_feat)
        return all_feat


class FreqFusionModule(nn.Module):
    def __init__(self, dim_out=1024, dim_mid=1024):
        super(FreqFusionModule, self).__init__()
        # conv
        dim_in = 3*64
        dim_mid = dim_mid
        div_ratio = 2

        self.stem_convs = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(), # n, 192, 28 , 28
            nn.Conv2d(dim_in, dim_mid, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(dim_mid),
            nn.ReLU(),  # 16x
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.channel_attn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_mid, dim_mid // div_ratio),
            nn.ReLU(),
            nn.Linear(dim_mid // div_ratio, dim_mid),
            nn.Sigmoid()
        )

        self.freq_pooling = nn.Sequential(
            nn.Conv2d(dim_mid, dim_mid, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(dim_mid),
            nn.ReLU(),
            nn.Conv2d(dim_mid, dim_mid, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
        )

        # self.fusion_conv = nn.Sequential(
        #     nn.Linear(2048 + dim_mid, dim_out),
        #     nn.ReLU(),
        #     nn.Linear(dim_out, dim_out)
        # )

        self.channel_attn[1].weight.data.normal_(0, 0.01)
        self.channel_attn[1].bias.data.fill_(0.0)
        self.channel_attn[3].weight.data.normal_(0, 0.01)
        self.channel_attn[3].bias.data.fill_(0.0)
        # self.fusion_conv[0].weight.data.normal_(0, 0.01)
        # self.fusion_conv[0].bias.data.fill_(0.0)
        # self.fusion_conv[2].weight.data.normal_(0, 0.01)
        # self.fusion_conv[2].bias.data.fill_(0.0)

    def forward(self, x_ycbcr, rgb_feat):
        """
        :param x: ycbcr -> dct -> x
        :return: frequency features
        """
        freq_feat = self.stem_convs(x_ycbcr)  # N, C,
        b, c, _, _ = freq_feat.size()
        freq_attn = self.channel_attn(freq_feat).view(b, c, 1, 1)
        freq_feat = freq_feat + freq_feat * freq_attn.expand_as(freq_feat)

        freq_feat = self.freq_pooling(freq_feat).flatten(start_dim=1)
        all_feat = torch.cat((rgb_feat, freq_feat), dim=1)
        # all_feat = self.fusion_conv(all_feat)
        return all_feat


class MultiScaleFusion(nn.Module):
    def __init__(self, cfg=None):
        super(MultiScaleFusion, self).__init__()
        self.output_block = [0, 3, 7, 11]
        self.convs = nn.ModuleList()
        for _ in range(len(self.output_block)):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(768, 256, kernel_size=(1,1), padding=(0,0), stride=(1,1)),
                    nn.BatchNorm2d(256),
                    nn.ReLU()
                )
            )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256 * len(self.output_block), 256 * len(self.output_block) // 2, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256 * len(self.output_block) // 2),
            nn.ReLU(),
            nn.Conv2d(256 * len(self.output_block)//2, 256 * len(self.output_block) // 2, kernel_size=(3, 3),
                      padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256 * len(self.output_block) // 2),
            nn.ReLU(),
            nn.Conv2d(256 * len(self.output_block) // 2, 256 * len(self.output_block), kernel_size=(1,1), padding=(0, 0), stride=(1,1)),
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten()
        )

        self.fusion_rgb = nn.Sequential(
            nn.Linear(1024 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

    def reshape_to_conv_feat(self, multi_scale_feats):
        outputs = []
        for feat in multi_scale_feats:
            reshape_feat = feat[:, 1:].permute(0, 2, 1).contiguous()
            _, C, M = reshape_feat.shape
            H, W = int(math.sqrt(M)), int(math.sqrt(M))
            reshape_feat = reshape_feat.view(_, C, H, W)
            outputs.append(reshape_feat)
        return outputs

    def forward(self, multi_scale_feats, rgb_feat):
        reshape_feats = self.reshape_to_conv_feat(multi_scale_feats)
        for _ in range(len(multi_scale_feats)):
            reshape_feats[_] = self.convs[_](reshape_feats[_])
        out = self.fusion_conv(torch.cat(reshape_feats, dim=1))

        out = self.fusion_rgb(torch.cat([rgb_feat, out], dim=1))
        return out


if __name__ == '__main__':
    m = FreqFusionModule()
    m = FreqFusionXFormerModule(dim_out=1024)
    # m = MultiScaleFusion()
    mfeat = torch.randn(2, 192, 28, 28)
    f = torch.randn(2, 1024)

    out = m(mfeat, f)
    print(out.size())
    # mfeats = [mfeat] *4
    # out = m(mfeats)
    # print(out.size())
    # x = torch.randn(2, 256, 8, 8)
    # feat = torch.randn(2, 1024)
    # y = m(x, feat)
    # print(y.size())

    import cv2
    import numpy as np
    # im = cv2.imread('/home/og/home/lqx/code/FaceAdaptation/datasets/test.png')
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    # y,cr,cb = im[:,:,0].astype(np.float),im[:,:,1].astype(np.float),im[:,:,2].astype(np.float)
    # y_dct, cr_dct, cb_dct = cv2.dct(y), cv2.dct(cr), cv2.dct(cb)

    # x = torch.randn(2, 3, 224, 224)
    # x2 = torch.randn(2, 1024)
    # # x_dct = dct_2d(x, norm='ortho')
    # x_dct = m(x, x2)

    print('done')

    #
    # image         ->  ycbcr          -> DCT on 8x8 block -> concat ycbcr
    # (3, 224, 224) ->  (3, 224, 224)  -> (3, 64, 28, 28)  -> (192, 28, 28)