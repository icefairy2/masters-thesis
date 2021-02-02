import functools

import torch
import torch.nn as nn

import resnet


def get_model(arch):
    if hasattr(resnet, arch):
        network = getattr(resnet, arch)
        return network(pretrained=True, num_classes=512)
    else:
        raise ValueError("Invalid Backbone Architecture")


class H3DWEncoder(nn.Module):
    def __init__(self, opt, mean_params):
        super(H3DWEncoder, self).__init__()
        self.two_branch = opt.two_branch
        self.mean_params = mean_params.clone().cuda()
        self.opt = opt

        relu = nn.ReLU(inplace=False)
        fc2 = nn.Linear(1024, 1024)
        regressor = nn.Linear(1024 + opt.total_params_dim, opt.total_params_dim)

        feat_encoder = [relu, fc2, relu]
        regressor = [regressor, ]
        self.feat_encoder = nn.Sequential(*feat_encoder)
        self.regressor = nn.Sequential(*regressor)

        self.main_encoder = get_model(opt.main_encoder)

    def forward(self, main_input):
        main_feat = self.main_encoder(main_input)
        feat = self.feat_encoder(main_feat)

        pred_params = self.mean_params
        for i in range(3):
            input_feat = torch.cat([feat, pred_params], dim=1)
            output = self.regressor(input_feat)
            pred_params = pred_params + output
        return pred_params
