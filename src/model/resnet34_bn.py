
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet34_BN(nn.Module):
    """
    Build a ResNet34 model as simsiam trunk.
    """
    def __init__(self, pretrained=False, syncBN=False):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(ResNet34_BN, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        base_encoder = models.__dict__["resnet34"]
        self.model = base_encoder(zero_init_residual=True, pretrained=pretrained)

        # if syncBN:
        #     self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # not used fc layer
        self.last_linear = self.model.fc
        self.model.fc = nn.Linear(self.model.fc.in_features, 128)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)
        return x
