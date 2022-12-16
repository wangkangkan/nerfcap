import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
from  matplotlib import pyplot as plt
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        self.annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = self.annots['cams']

        num_cams = len(self.cams['K'])

        self.subsequence_framenumber = 20
        test_view = cfg.test_view
        view = cfg.training_view if self.split == 'train' else test_view
        if len(view) == 0:
            view = [0]
        self.viewnum = len(view)

        # all sequences are concatenated in one
        begin_ith_frame = cfg.begin_ith_frame
        i = 0
        i = i + begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
        if cfg.test_novel_pose:
            i = (i + cfg.num_train_frame) * i_intv
            ni = cfg.num_novel_pose_frame
            if self.human == 'CoreView_390':
                i = 0
 
        self.multiviewims = np.array([
            np.array(ims_data['ims'])
            for ims_data in self.annots['ims'][i:i + ni][::1]
        ])

        self.multiviewcam_inds = np.array([
            np.arange(len(ims_data['ims']))
            for ims_data in self.annots['ims'][i:i + ni][::1]
        ])
        
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in self.annots['ims'][i:i + ni][::i_intv]
        ]).ravel()
        
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in self.annots['ims'][i:i + ni][::i_intv]
        ]).ravel()

        self.num_cams = len(view)

        self.nrays = cfg.N_rand

        joints = np.load(os.path.join(self.data_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        parents = np.load(os.path.join(self.data_root, 'parents.npy'), allow_pickle=True)
        self.parents = np.array(parents)

    def setsequenceidx(self, seqidx):
        test_view = [0]
        view = cfg.training_view if self.split == 'train' else test_view
        if len(view) == 0:
            view = [0]
            
        # all sequences are concatenated in one sequence
        cfg.begin_ith_frame = seqidx*cfg.num_train_frame+400#
        i = 0
        i = i + cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
        if cfg.test_novel_pose:
            i = (i + cfg.num_train_frame) * i_intv
            ni = cfg.num_novel_pose_frame
            if self.human == 'CoreView_390':
                i = 0
 
        self.multiviewims = np.array([
            np.array(ims_data['ims'])
            for ims_data in self.annots['ims'][i:i + ni][::i_intv]
        ])

        self.multiviewcam_inds = np.array([
            np.arange(len(ims_data['ims']))
            for ims_data in self.annots['ims'][i:i + ni][::i_intv]
        ])
        
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in self.annots['ims'][i:i + ni][::i_intv]
        ]).ravel()
        
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in self.annots['ims'][i:i + ni][::i_intv]
        ]).ravel()

        self.num_cams = len(view)

        self.nrays = cfg.N_rand
    
    def set_pytorch3d_intrinsic_matrix(self, K, H, W):
        fx = -K[0, 0] * 2.0 / W
        fy = -K[1, 1] * 2.0 / H
        px = -(K[0, 2] - W / 2.0) * 2.0 / W
        py = -(K[1, 2] - H / 2.0) * 2.0 / H
        K = [
            [fx, 0, px, 0],
            [0, fy, py, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
        K = np.array(K)
        return K
    
    def get_mask(self, index):
        msk_path = os.path.join('../../deepcap_dataset/Magdalena/training', 'foregroundSegmentation') + self.ims[index][6:]

        mask = imageio.imread(msk_path)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        msk = mask / 255

        return msk
        
    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, 'vertices', '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)
        vert = xyz

        bound_thr = 0.05

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        if cfg.big_box:
            min_xyz -= bound_thr
            max_xyz += bound_thr
        else:
            min_xyz[2] -= bound_thr
            max_xyz[2] += bound_thr

        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, 'smpl_params', '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= bound_thr
            max_xyz += bound_thr
        else:
            min_xyz[2] -= bound_thr
            max_xyz[2] += bound_thr
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)

        return coord, out_sh, can_bounds, bounds, Rh, Th, vert, A

    def read_multiviewimgs(self, img, cam_ind, index):

        multiviewimgs = []
        for viewidx in cfg.training_view:
            img_path = os.path.join('../../deepcap_dataset/Magdalena/training', self.multiviewims[index][viewidx])
            img = imageio.imread(img_path).astype(np.float32) / 255.
            msk_path = os.path.join('../../deepcap_dataset/Magdalena/training', 'foregroundSegmentation') + self.multiviewims[index][viewidx][6:]

            mask = imageio.imread(msk_path)
            _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
            msk = mask / 255
            if np.sum(msk)>0:
                img[msk == 0] = 0
            img = cv2.resize((img-0.5)/0.5, (224, 224), interpolation=cv2.INTER_AREA)
            multiviewimgs.append(img[None,...])
        multiviewimgs = np.concatenate(multiviewimgs)
         
        return multiviewimgs
    
    def read_multiviewcampara(self, index):

        multiviewK = []
        multiviewRT = []

        for viewidx in cfg.training_view:
            cam_ind = self.multiviewcam_inds[index][viewidx]
            K = np.array(self.cams['K'][cam_ind])
            
            R = np.array(self.cams['R'][cam_ind])
            T = np.array(self.cams['T'][cam_ind]) / 1000.
            RT = np.concatenate([R, T], axis=1).astype(np.float32)
        
            multiviewK.append(K[None,...])
            multiviewRT.append(RT[None,...])  
            
        multiviewK = np.concatenate(multiviewK)
        multiviewRT = np.concatenate(multiviewRT)
        return multiviewK, multiviewRT
        
    def __getitem__(self, index):
        img_path = os.path.join('../../deepcap_dataset/Magdalena/training', self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk = self.get_mask(index)
        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.
        RT = np.concatenate([R, T], axis=1).astype(np.float32)

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        if cfg.mask_bkgd and np.sum(msk)>0:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1

        K[:2] = K[:2] * cfg.ratio

        # make sure index starts at zero
        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path).split('_')[4][:-4])
            frame_index = i

        coord, out_sh, can_bounds, bounds, Rh, Th, vert, A = self.prepare_input(i)

        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray(
            img, msk, K, R, T, can_bounds, self.nrays, self.split)
        
        multiviewimgs = self.read_multiviewimgs(img, cam_ind, frame_index - cfg.begin_ith_frame)
        multiviewK, multiviewRT = self.read_multiviewcampara(frame_index - cfg.begin_ith_frame)
        ret = {
            'coord': coord_,
            'out_sh': out_sh,
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box,
            'msk': msk,
            'vert': vert,
            'multiviewimgs': multiviewimgs,
            'A': A,
            'multiviewK': multiviewK.astype(np.float32), 
            'multiviewRT': multiviewRT
        }
        pytorch3d_K = self.set_pytorch3d_intrinsic_matrix(K, cfg.H, cfg.W)

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = frame_index - cfg.begin_ith_frame

        if cfg.test_novel_pose:
            latent_index = cfg.num_train_frame - 1
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': cam_ind,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        R0 = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {'R0_snap': R0, 'Th0_snap': Th, 'K': K.astype(np.float32), 'RT': RT, 'pytorch3d_K': pytorch3d_K}
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
