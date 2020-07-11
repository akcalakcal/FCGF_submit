import os
import sys
import numpy as np
import argparse
import logging
import open3d as o3d

sys.path.append("/home/akin/workspace/workspace_applications/Deep_3D_Search/FCGF_submit/")

from model import load_model
from util.misc import extract_features

'''

from lib.timer import Timer, AverageMeter

from util.file import ensure_dir, get_folder_list, get_file_list
from util.trajectory import read_trajectory, write_trajectory
from util.pointcloud import make_open3d_point_cloud, evaluate_feature_3dmatch
from scripts.benchmark_util import do_single_pair_matching, gen_matching_pair, gather_results
'''

import torch

from pathlib import Path

import MinkowskiEngine as ME

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class FeatureExtractor:
    def __init__(self, modelpath):
        self.modelpath = modelpath
        self.with_cuda = True

    def extract(self, pointcloudpath, targetpath):
        print('control output')

        FeatureContainer={}

        device = torch.device('cuda' if self.with_cuda else 'cpu')

        checkpoint = torch.load(self.modelpath)
        config = checkpoint['config']

        num_feats = 1
        Model = load_model(config.model)
        model = Model(
            num_feats,
            config.model_n_out,
            bn_momentum=0.05,
            normalize_feature=config.normalize_feature,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        model = model.to(device)
        voxel_size = config.voxel_size

        with torch.no_grad():
            fi = pointcloudpath
            pcd = o3d.io.read_point_cloud(fi)

            xyz_down, feature = extract_features(
                model,
                xyz=np.array(pcd.points),
                rgb=None,
                normal=None,
                voxel_size=voxel_size,
                device=device,
                skip_check=True)

        #FeatureContainer.update('points', np.array(pcd.points))
        #FeatureContainer.update('xyz', xyz_down)
        #FeatureContainer.update('feature', feature.detach().cpu().numpy())

        #return FeatureContainer

            fi_base = os.path.basename(fi)
            print('fi_base = ', fi_base)
            target_path = targetpath
            path_save_feature = target_path + '/' + fi_base.split('.')[0] + ".npz"
            print('path_save_feature = ', path_save_feature)

            np.savez_compressed(
                path_save_feature,
                points=np.array(pcd.points),
                xyz=xyz_down,
                feature=feature.detach().cpu().numpy())

            return  path_save_feature

