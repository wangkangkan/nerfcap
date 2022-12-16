import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from lib.utils.blend_utils import *
from . import embedder
from lib.utils import net_utils
import os
import numpy as np
from . import Resnet
from . import feature_network

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Resnet.load_Res50Model()
        self.encoder_appearance = feature_network.ResUNet(coarse_out_ch=128, fine_out_ch=128, coarse_only=True)
        
        self.occupancy_network = OccupancyNetwork()
        self.color_network = ColorNetwork()
        self.deformation_network = DeformationNetwork()

        net_utils.load_network(self, 'trained_model/if_nerf/init_occ_mg/', strict=False)

        activation_func = nn.ELU(inplace=True)       
        self.multiview_fc = nn.Sequential(nn.Linear(128+3, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid())

    def extract_multiviewptsfeature(self, wpts, sp_input, appearancelatent):
        multiviewRT = sp_input['multiviewRT'].squeeze(0)
        multiviewK = sp_input['multiviewK'].squeeze(0)

        num_views = sp_input['multiviewRT'].size()[1]  # view number
        pts_size = wpts.size()[1]
        wpts = wpts.repeat(num_views, 1, 1)

        localwpts = torch.matmul(wpts, multiviewRT[:,:,:3].transpose(1, 2)) + multiviewRT[:,:,-1].unsqueeze(1).repeat(1, pts_size, 1)

        coord = torch.matmul(localwpts, multiviewK.transpose(1, 2))  # B*P*2
        coord[..., 0] = torch.div(coord[..., 0], torch.clamp(coord[..., 2], min=1e-8))
        coord[..., 1] = torch.div(coord[..., 1], torch.clamp(coord[..., 2], min=1e-8))

        coord[..., 0] = 2 * torch.true_divide(coord[..., 0], cfg.W-1.) - 1.
        coord[..., 1] = 2 * torch.true_divide(coord[..., 1], cfg.H-1.) - 1.
        
        coordidx = coord[..., 0:2][:, None]

        out = torch.nn.functional.grid_sample(appearancelatent, coordidx, padding_mode='border', align_corners=True)
        out = out.permute(0, 2, 3, 1)

        multiviewpts_appearancefeature = out.view(num_views, pts_size, -1)  # out[:,1,P,:]

        multiviewrays_o = -torch.matmul(multiviewRT[:,:,:3].transpose(1, 2), multiviewRT[:,:,-1].unsqueeze(-1))
        rays_o = -torch.matmul(sp_input['RT'][:,:,:3].transpose(1, 2), sp_input['RT'][:,:,-1].unsqueeze(-1))
        rays_o = rays_o.repeat(num_views, 1, 1)
        
        ray2tar_pose = (rays_o.transpose(1, 2).repeat(1, pts_size, 1) - wpts)
        ray2tar_pose /= (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2train_pose = (multiviewrays_o.transpose(1, 2).repeat(1, pts_size, 1) - wpts)
        ray2train_pose /= (torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, multiviewpts_appearancefeature], dim=-1)

        weight = self.multiview_fc(ray_diff)
        weight = weight.permute(1,0,2)  # P*N*1
        weight = weight /(torch.sum(weight, dim=1, keepdim=True) + 1e-6)
        multiviewpts_appearancefeature = multiviewpts_appearancefeature.permute(1,0,2)  # P*N*C
        pts_appearancefeature = torch.sum(multiviewpts_appearancefeature*weight, dim=1, keepdim=True)  # P*1*C
        pts_appearancefeature = pts_appearancefeature.permute(1,0,2)  # 1*P*C

        return pts_appearancefeature
        
    def calculate_density_color_deformation(self, wpts, viewdir, sp_input, appearancelatent):
        pts_appearancefeature = self.extract_multiviewptsfeature(wpts, sp_input, appearancelatent)

        canwpts = self.deformation_network.inversedeforming_samplepoints_LBS_graphdeform(wpts, sp_input)

        light_pts = embedder.xyz_embedder(canwpts)
        alpha, h = self.occupancy_network(light_pts)
        rgb = self.color_network(light_pts, viewdir, sp_input, pts_appearancefeature)
        
        raw = torch.cat([rgb, alpha], -1)

        return raw

    def forward(self, sp_input, grid_coords, viewdir, light_pts):
        __import__('ipdb').set_trace()
        
class DeformationNetwork(nn.Module):
    def __init__(self):
        super(DeformationNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        templatesmpl_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/vpersonalshape.txt')
        templatesmpl = np.loadtxt(templatesmpl_path)
        templatesmpl = templatesmpl
        self.templatesmpl = torch.Tensor(templatesmpl).to(self.device)

        # loading template deformation graph
        templateshape_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/templateshapeT.txt')
        templateshape = np.loadtxt(templateshape_path)
        templateshape = templateshape
        self.templateshape = torch.Tensor(templateshape).to(self.device)
        self.vtnum = self.templateshape.size(0)
        modelnodepos_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/modelnodepos.txt')
        modelnodepos = np.loadtxt(modelnodepos_path)
        modelnodepos = modelnodepos / 1000.0
        self.modelnodepos = torch.Tensor(modelnodepos).to(self.device)
        modelnodenormal_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/modelnodenormal.txt')
        modelnodenormal = np.loadtxt(modelnodenormal_path)
        self.modelnodenormal = torch.Tensor(modelnodenormal).to(self.device)
        self.modelnodenum = self.modelnodepos.size(0)
        modelnodeedge_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/modelnodeedge.txt')
        modelnodeedge = np.loadtxt(modelnodeedge_path) - 1
        self.modelnodeedge = torch.LongTensor(modelnodeedge).to(self.device)
        self.modelnodeedgenum = self.modelnodeedge.size(1)
        modelnode_edgeweight_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/modelnode_edgeweight.txt')
        modelnode_edgeweight = np.loadtxt(modelnode_edgeweight_path)
        self.modelnode_edgeweight = torch.Tensor(modelnode_edgeweight).to(self.device)
        modelvertnode_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/modelvert_node.txt')
        modelvert_node = np.loadtxt(modelvertnode_path) - 1
        self.modelvert_node = torch.LongTensor(modelvert_node).to(self.device)
        modelvertnodeweight_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/modelvert_nodeweight.txt')
        modelvert_nodeweight = np.loadtxt(modelvertnodeweight_path)
        self.modelvert_nodeweight = torch.Tensor(modelvert_nodeweight).to(self.device)
        self.modelvert_nodenum = self.modelvert_node.size(1)

        hipidx_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/hipidx.txt')
        hipidx = np.loadtxt(hipidx_path) - 1
        hipidx = torch.LongTensor(hipidx).to(self.device)
        skirtidx_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/skirtidx.txt')
        skirtidx = np.loadtxt(skirtidx_path) - 1
        skirtidx = torch.LongTensor(skirtidx).to(self.device)

        bw = np.load(os.path.join(cfg.train_dataset.data_root, 'bw.npy'), allow_pickle=True)
        bw = torch.Tensor(bw).to(self.device)
        ptsdist = torch.cdist(self.templateshape, self.templatesmpl, p=2)
        minptsdist = torch.min(ptsdist, 1)
        minptsdistvalue = torch.squeeze(minptsdist[0], -1)
        nnvidx = torch.squeeze(minptsdist[1], -1)  # P
        self.bw = bw[nnvidx, :]

        hipvt = self.templatesmpl[hipidx, :]
        ptsdisthip = torch.cdist(self.templateshape, hipvt, p=2)
        minptshipdist = torch.min(ptsdisthip, 1)
        minptsdisthipidx = torch.squeeze(minptshipdist[1], -1)
        bwhip = bw[hipidx[minptsdisthipidx], :]

        bwhip = bw[hipidx, :]
        bwhipmean = torch.mean(bwhip, 0)
        t = minptsdistvalue > 0.1
        self.bw[skirtidx, :] = bwhipmean
        self.bw = self.bw[None, ...]

        D = 8
        self.deformskips = [4]
        defW = 128
        layers = [nn.Linear(128, defW)]  # node coding + latent code
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = defW
            if i in self.deformskips:
                in_channels += 128
            layers += [layer(in_channels, defW)]

        self.deformpara_linears = nn.ModuleList(layers)
        self.deformpara_finallinear = nn.Linear(defW, self.modelnodenum * 6)

    def deformationsmoothloss(self):
        #smooth constrain loss
        repmodelnodepos = self.modelnodepos.unsqueeze(1).repeat(1, self.modelnodeedgenum, 1)  # nodenum*edgenum*3
        repmodelnodepos = repmodelnodepos.view([-1, 3])  # (nodenum*edgenum)*3

        self.modelnodeedge = self.modelnodeedge.view([-1])  # (nodenum*edgenum)
        relativepos = repmodelnodepos - self.modelnodepos[self.modelnodeedge, :]  # (nodenum*edgenum)*3
        relativepos = relativepos[None, ..., None]
        deformrelativepos = torch.matmul(self.deformation_affine[:, self.modelnodeedge, :, :],
                                         relativepos)  # B*(nodenum*edgenum)*3*3, #B*(nodenum*edgenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.modelnodepos[self.modelnodeedge,
                                                    :][None, ...] + self.deformation_transl[:, self.modelnodeedge, :]
        deformpos = deformpos.view(-1, self.modelnodenum*self.modelnodeedgenum, 3)  # B*(nodenum*edgenum)*3
        repnodetransl = self.deformation_transl.unsqueeze(2).repeat(1, 1, self.modelnodeedgenum, 1)  # B*nodenum*edgenum*3
        repnodetransl = repnodetransl.view([-1, self.modelnodenum*self.modelnodeedgenum, 3])  # B*(nodenum*edgenum)*3

        smoothpos = deformpos - (repmodelnodepos[None, ...] + repnodetransl)
        smoothpos = smoothpos.view(-1, self.modelnodenum, self.modelnodeedgenum, 3)  # B*nodenum*edgenum*3
        smoothpos = smoothpos**2

        smoothloss = torch.sum(smoothpos)
        return smoothloss

    def batch_rodrigues(self, rot_vecs, epsilon=1e-8, dtype=torch.float32):

        batch_size = rot_vecs.shape[0]
        device = rot_vecs.device

        angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
        rot_dir = rot_vecs / angle

        cos = torch.unsqueeze(torch.cos(angle), dim=1)
        sin = torch.unsqueeze(torch.sin(angle), dim=1)

        # Bx1 arrays
        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

        zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
            .view((batch_size, 3, 3))

        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)

        t = torch.bmm(rot_dir.unsqueeze(2), rot_dir.unsqueeze(1))
        rot_mat = cos * ident + (1 - cos) * t + sin * K
        return rot_mat

    def predicting_deformation(self, sp_input, latent):
        h = latent
        for i, l in enumerate(self.deformpara_linears):
           h = self.deformpara_linears[i](h)
           h = F.relu(h)
           if i in self.deformskips:
               h = torch.cat([latent, h], -1)

        h = self.deformpara_finallinear(h)
        h = h.view(-1,self.modelnodenum,6)

        deformation_affine, deformation_transl = torch.split(h, [3, 3], dim=-1)
        deformation_rotate = self.batch_rodrigues(deformation_affine.view([-1, 3]))  # (B*nodenum)*3->(B*nodenum)*3*3

        self.deformation_affine = deformation_rotate.view(-1, self.modelnodenum, 3, 3)
        self.deformation_transl = deformation_transl.view(-1, self.modelnodenum, 3)

        return self.deformation_affine, self.deformation_transl

    def deformingtemplate(self):
        # deforming template on all nodes once
        reptemplateshape = self.templateshape.unsqueeze(1).repeat(1, self.modelvert_nodenum, 1)  # vtnum*nodenum*3
        reptemplateshape = reptemplateshape.view([-1, 3])  # (vtnum*nodenum)*3
        self.modelvert_node =self.modelvert_node.view([-1])  # (vtnum*nodenum)
        relativepos = reptemplateshape - self.modelnodepos[self.modelvert_node, :]  # (vtnum*nodenum)*3
        relativepos = relativepos[None, ..., None]
        deformrelativepos = torch.matmul(self.deformation_affine[:, self.modelvert_node, :, :], relativepos)  # B*(vtnum*nodenum)*3*3, #B*(vtnum*nodenum)*3*1
        deformpos = deformrelativepos.squeeze(-1) + self.modelnodepos[self.modelvert_node,
                                                   :][None, ...] + self.deformation_transl[:, self.modelvert_node, :]  # B*(vtnum*nodenum)*3
        deformpos = deformpos.view(-1,self.vtnum,self.modelvert_nodenum,3)  # B*vtnum*nodenum*3
        weighteddeformpos = deformpos*self.modelvert_nodeweight[None, ..., None]  # nodeweight: B*vtnum*nodenum*1
        weighteddeformpos = torch.sum(weighteddeformpos, dim=2)  # B*vtnum*1*3
        self.deformedverts = weighteddeformpos.squeeze(2)  # B*vtnum*3

        return self.deformedverts

    def deformingtemplate_graphdeform_LBS(self, sp_input):

        # embedded deformation on template shape in T pose
        graphdeformedverts = self.deformingtemplate()

        # deforming with LBS of SMPL further
        sh = graphdeformedverts.shape
        A = torch.bmm(self.bw, sp_input['A'].view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)
        R = A[..., :3, :3]  # not including global rotation
        pts = torch.sum(R * graphdeformedverts[:, :, None], dim=3)
        pts = pts + A[..., :3, 3]
        self.deformedverts = torch.matmul(pts, sp_input['R'].transpose(1, 2)) + sp_input['Th']

        return self.deformedverts
        
    def inversedeforming_samplepoints_LBS_graphdeform(self, wpts, sp_input):
        #inversely deforming sample points to canonical frame
        ptsnum = wpts.size(1)
        ptsdist = torch.cdist(wpts, self.deformedverts, p=2)
        nnvidx = torch.squeeze(torch.min(ptsdist, 2)[1], -1)  # B*P

        templatevtnum = self.deformedverts.size(1)

        # world points to posed points
        pts = torch.matmul(wpts - sp_input['Th'], sp_input['R'])

        # transform points from the pose space to the T pose
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(wpts.size(0))]
        idx = torch.cat(idx).to(self.device)
        bwidx = nnvidx.view(-1) + idx
        bw1 = self.bw.view(-1, 24)  #
        selectbw = bw1[bwidx.long(), :]
        selectbw = selectbw.view(-1, ptsnum, 24)

        sh = pts.shape
        A = torch.bmm(selectbw, sp_input['A'].view(sh[0], 24, -1))
        A = A.view(sh[0], -1, 4, 4)  # A: n_batch, 24, 4, 4
        pts = pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3])
        pts = torch.sum(R_inv * pts[:, :, None], dim=3)

        # inverse embedded deformation
        repwpts = pts.unsqueeze(2).repeat(1, 1, self.modelvert_nodenum, 1)  # B*vtnum*nodenum*3
        self.modelvert_node = self.modelvert_node.view([-1,self.modelvert_nodenum])
        ptsnode = self.modelvert_node[nnvidx.view([-1]),:]  # (B*P)*nodenum
        ptsnode = ptsnode.view([-1])  # (B*P*nodenum); the influence nodes for each pt

        sh = ptsnum * self.modelvert_nodenum
        idx = [torch.full([sh], i * self.modelnodenum, dtype=torch.long) for i in range(pts.size(0))]
        idx = torch.cat(idx).to(self.device)
        ptsnodeidx = ptsnode + idx  # batch idx of the influence nodes, used for retrieving deformation (affine and transl) of each batch sample
        ptsnodeidx = ptsnodeidx.long()
        deformtransl = self.deformation_transl.view(-1, 3)  # (B*modelnodenum)*3
        selectdeformtransl = deformtransl[ptsnodeidx, :]
        selectdeformtransl = selectdeformtransl.view(-1, ptsnum,self.modelvert_nodenum, 3)  # B*(vtnum*nodenum)*3

        repwpts = repwpts.view([-1, ptsnum,self.modelvert_nodenum, 3])
        nodepos = self.modelnodepos[ptsnode, :].view([-1, ptsnum, self.modelvert_nodenum, 3])
        relativepos = repwpts - nodepos - selectdeformtransl  # B*vtnum*nodenum*3

        deformaffine = self.deformation_affine.view(-1, 3, 3)
        selectdeformaffine = deformaffine[ptsnodeidx, :, :]  # (B*vtnum*nodenum)*3*3
        selectdeformaffine = selectdeformaffine.view(-1, ptsnum, self.modelvert_nodenum, 3, 3)  # B*vtnum*nodenum*3*3
        deformednodepos = torch.matmul(selectdeformaffine, nodepos[..., None])  # B*vtnum*nodenum*3*1
        relativepos = relativepos + deformednodepos.squeeze(-1)  # B*vtnum*nodenum*3

        ptsnodeweight = self.modelvert_nodeweight[nnvidx.view([-1]), :]  # (B*vtnum)*nodenum
        ptsnodeweight = ptsnodeweight.view([-1, ptsnum, self.modelvert_nodenum,1])
        weightedrelativepts = relativepos * ptsnodeweight  # B*vtnum*nodenum*3
        weightedrelativepts = torch.sum(weightedrelativepts, dim=2)  # B*vtnum*1*3
        weightedrelativepts = weightedrelativepts.squeeze(2)

        weighteddeformaffine = selectdeformaffine * ptsnodeweight[...,None]
        weighteddeformaffine = torch.sum(weighteddeformaffine, dim=2)  # B*vtnum*1*3*3
        weighteddeformaffine = weighteddeformaffine.squeeze(2)
        a = weighteddeformaffine.view(-1, 3, 3)  # (B*vtnum)*3*3
        c = a.inverse()
        inversedeformaffine = c.view(-1, ptsnum, 3, 3)  # B*vtnum*3*3

        deformpts = torch.matmul(inversedeformaffine, weightedrelativepts[..., None])  # B*vtnum*3*1
        deformpts = deformpts.squeeze(-1)  # B*vtnum*3

        return deformpts

class OccupancyNetwork(nn.Module):
    def __init__(self):
        super(OccupancyNetwork, self).__init__()

        self.actvn = nn.ReLU()

        self.skips = [4]
        D = 8
        W = 256
        input_ch = 63
        input_ch_views = 27
        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = W
            if i in self.skips:
                in_channels += input_ch
            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)
        self.alpha_linear = nn.Linear(W, 1)
        self.alpha_linear.bias.data.fill_(0.693)

    def forward(self, light_pts):
        h = light_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([light_pts, h], -1)
        alpha = self.alpha_linear(h)
        occupancy = 1 - torch.exp(-torch.relu(alpha))
        return occupancy, h

class ColorNetwork(nn.Module):
    def __init__(self):
        super(ColorNetwork, self).__init__()

        input_ch = 63
        input_ch_views = 27
        D = 8
        W = 256

        self.skips = [4]

        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            layer = nn.Linear
            in_channels = W
            if i in self.skips:
                in_channels += input_ch
            layers += [layer(in_channels, W)]
        self.pts_linears = nn.ModuleList(layers)

        self.views_linears = nn.ModuleList([nn.Linear(W, W // 2)])
        self.feature_linear = nn.Linear(W, W)
        self.rgb_linear = nn.Linear(W // 2, 3)
        self.latent_fc = nn.Linear(384, 256)

    def forward(self, light_pts, viewdir, sp_input, latent):

        input_h = light_pts
        h = input_h
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_h, h], -1)
        
        features = self.feature_linear(h)

        features = torch.cat((features, latent), dim=2)
        features = self.latent_fc(features)

        h = features
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)

        return rgb