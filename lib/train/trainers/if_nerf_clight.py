import torch.nn as nn
import torch
from lib.networks.renderer import if_clight_renderer_occupancy

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = if_clight_renderer_occupancy.Renderer(self.net)

        for param in self.net.occupancy_network.parameters():
            param.requires_grad = False

        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.acc_crit = torch.nn.functional.smooth_l1_loss
        self.msk2mse = lambda x, y: torch.mean((x - y) ** 2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, batch, epoch):
        ret = self.renderer.render_deformation(batch, is_test=False)

        scalar_stats = {}
        loss = 0

        mask = batch['mask_at_box']
        img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
       
        scalar_stats.update({'img_loss': img_loss})
        loss += 1.0 *img_loss

        loss += 0.005 * self.renderer.smoothloss
        scalar_stats.update({'smoothloss': self.renderer.smoothloss})

        if 'rgb0' in ret:
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
