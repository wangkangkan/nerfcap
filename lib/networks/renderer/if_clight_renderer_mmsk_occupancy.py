import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from . import if_clight_renderer_occupancy


class Renderer(if_clight_renderer_occupancy.Renderer):
    def __init__(self, net):
        super(Renderer, self).__init__(net)

    def get_density_color(self, wpts, viewdir, inside, raw_decoder):
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch, n_pixel * n_sample, -1)
        wpts = wpts[inside][None]
        viewdir = viewdir[inside][None]
        full_raw = torch.zeros([n_batch, n_pixel * n_sample, 4]).to(wpts)
        if inside.sum() == 0:
            return full_raw

        raw = raw_decoder(wpts, viewdir)
        full_raw[inside] = raw[0]

        return full_raw

    def get_pixel_value(self, ray_o, ray_d, near, far, sp_input, batch, deformedverts, appearancecode):#
        # sampling points along camera rays
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)
        inside = self.prepare_inside_pts(wpts, batch)

        n_batch, n_pixel, n_sample = wpts.shape[:3]
        ptsdist = torch.cdist(wpts.view(n_batch,-1,3), deformedverts, p=2)#,
        nndist = torch.squeeze(torch.min(ptsdist, 2)[0], -1)  # B*P
        ptsnearsurfacetag = torch.where(nndist > 0.03, torch.zeros_like(nndist), torch.ones_like(nndist))
        
        # viewing direction
        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)

        raw_decoder = lambda x_point, viewdir_val: self.net.calculate_density_color_deformation(
            x_point, viewdir_val, sp_input, appearancecode)#

        # compute the color and density
        wpts_raw = self.get_density_color(wpts, viewdir, inside, raw_decoder)

        # volume rendering for wpts
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        raw = wpts_raw.reshape(-1, n_sample, 4)
        
        raw[...,3] = raw[...,3]*ptsnearsurfacetag.reshape(-1, n_sample)
        
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d, cfg.raw_noise_std, cfg.white_bkgd)

        ret = {
            'rgb_map': rgb_map.view(n_batch, n_pixel, -1),
            'disp_map': disp_map.view(n_batch, n_pixel),
            'acc_map': acc_map.view(n_batch, n_pixel),
            'weights': weights.view(n_batch, n_pixel, -1),
            'depth_map': depth_map.view(n_batch, n_pixel)
        }

        return ret
