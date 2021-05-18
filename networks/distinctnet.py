"""
DistinctNet class.
Run `python -m networks.distinctnet` for a dummy test.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torchvision.models.segmentation.deeplabv3 import ASPP

from networks.decoder import DeepLabDecoder
from networks.correlation import CorrelationLayer
from networks.convlstm import ConvLSTMLayer


class DistinctNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # determine forward type depending on number of output channels
        # 2: motion, default forward
        # 4: semantic segmentation
        if cfg.get("OUT_CH", 2) == 2:
            self._forward_type = 'motion'
        else:
            self._forward_type = 'semantics'

        backbone = resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True])

        # siamese / encoder
        self.intermediate = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)

        # best option is to merge after second layer
        # configure these lines if you want alternative merge options
        self.siamese = backbone.layer2
        self.encoder = nn.Sequential(backbone.layer3, backbone.layer4)

        # correlation layer
        self.corr_layer = CorrelationLayer()

        # reduction layer
        self.reduction_layer = nn.Sequential(
            nn.Conv2d(4784+1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # aspp
        self.aspp = ASPP(2048, [12, 24, 36])

        # decoder
        self.decoder = DeepLabDecoder(num_classes=cfg.get("OUT_CH", 2))

        # recurrent layers, if any
        self.recurrent = cfg.get("RECURRENT", False)
        if self.recurrent:
            self.rec_after_aspp = ConvLSTMLayer(in_ch=256, hidden_ch=[], k_size=3, padding=1, norm=True)
            self.decoder.last_conv = ConvLSTMLayer(in_ch=256+48, hidden_ch=[256, 256], out_ch=cfg.get("OUT_CH", 2), k_size=(3, 3, 1), padding=(1, 1, 0), norm=(True, True, False))

    def forward(self, data):
        input_shape = data['im1'].shape[-2:]

        if self._forward_type == 'motion' and not self.recurrent:  # train all layers
            # encoder
            im1_int = self.intermediate(data['im1'])
            im2_int = self.intermediate(data['im2'])

            # further encode
            im1_siam = self.siamese(im1_int)
            im2_siam = self.siamese(im2_int)

            # correlate
            corr = self.corr_layer(im1_siam, im2_siam).contiguous()
        else:  # freeze up to correlation layer
            with torch.no_grad():
                # encoder
                im1_int = self.intermediate(data['im1'])
                im2_int = self.intermediate(data['im2'])

                # further encode
                im1_siam = self.siamese(im1_int)
                im2_siam = self.siamese(im2_int)

                # correlate
                corr = self.corr_layer(im1_siam, im2_siam).contiguous()

        if not self.recurrent:
            # reduce
            reduced = self.reduction_layer(torch.cat((corr, im1_siam, im2_siam), dim=1))

            # further encode
            encoded = self.encoder(reduced)

            # aspp
            dec_inp = self.aspp(encoded)
        else:
            # freeze up to after aspp
            with torch.no_grad():
                # reduce
                reduced = self.reduction_layer(torch.cat((corr, im1_siam, im2_siam), dim=1))

                # further encode
                encoded = self.encoder(reduced)

                # aspp
                dec_inp = self.aspp(encoded)
            dec_inp = self.rec_after_aspp(dec_inp, data_str=data['path'])

        # decode
        out = self.decoder(dec_inp, im1_int, data_str=data['path'])
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=True)

        return {'pred': out}


if __name__ == '__main__':
    import torch
    inp = torch.randn(1, 3, 414, 736).cuda()

    print(f"Motion forward")
    cfg = {
        "OUT_CH": 2
    }
    net = DistinctNet(cfg=cfg).cuda().eval()
    with torch.no_grad():
        out = net({'im1': inp, 'im2': inp, 'path': torch.tensor([ord(c) for c in 'foo.bar'])})
    print(f"  Prediction shape: {out['pred'].shape}")

    print(f"Semantic forward")
    cfg = {
        "OUT_CH": 4
    }
    net = DistinctNet(cfg=cfg).cuda().eval()
    with torch.no_grad():
        out = net({'im1': inp, 'im2': inp, 'path': torch.tensor([ord(c) for c in 'foo.bar'])})
    print(f"  Prediction shape: {out['pred'].shape}")

    print(f"Motion + Recurrent forward")
    cfg = {
        "OUT_CH": 2,
        "RECURRENT": True
    }
    net = DistinctNet(cfg=cfg).cuda().eval()
    with torch.no_grad():
        out = net({'im1': inp, 'im2': inp, 'path': torch.tensor([ord(c) for c in 'foo.bar']).unsqueeze(0)})
    print(f"  Prediction shape: {out['pred'].shape}")

    print(f"Semantic + Recurrent forward")
    cfg = {
        "OUT_CH": 4,
        "RECURRENT": True
    }
    net = DistinctNet(cfg=cfg).cuda().eval()
    with torch.no_grad():
        out = net({'im1': inp, 'im2': inp, 'path': torch.tensor([ord(c) for c in 'foo.bar']).unsqueeze(0)})
    print(f"  Prediction shape: {out['pred'].shape}")
