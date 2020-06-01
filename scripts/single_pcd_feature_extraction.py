"""
A collection of unrefactored functions.
"""
import os
import sys
import numpy as np
import argparse
import logging
import open3d as o3d

sys.path.append("/vol/research/ucdatasets/MPSD/akin_tmp/src/FCGF/")

from lib.timer import Timer, AverageMeter

from util.misc import extract_features

from model import load_model
from util.file import ensure_dir, get_folder_list, get_file_list
from util.trajectory import read_trajectory, write_trajectory
from util.pointcloud import make_open3d_point_cloud, evaluate_feature_3dmatch
from scripts.benchmark_util import do_single_pair_matching, gen_matching_pair, gather_results

import torch

from pathlib import Path



import MinkowskiEngine as ME

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def main(args):

    target_dir = Path(args.target)
    target_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if args.with_cuda else 'cpu')

    checkpoint = torch.load(args.model)
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

        fi = args.source_file
        pcd = o3d.io.read_point_cloud(fi)

        xyz_down, feature = extract_features(
                  model,
                  xyz=np.array(pcd.points),
                  rgb=None,
                  normal=None,
                  voxel_size=voxel_size,
                  device=device,
                  skip_check=True)

        fi_base = os.path.basename(fi)
        print('fi_base = ', fi_base)
        target_path = args.target
        path_save_feature = target_path + fi_base.split('.')[0] + ".npz"
        print('path_save_feature = ', path_save_feature)

        np.savez_compressed(
            path_save_feature,
            points=np.array(pcd.points),
            xyz=xyz_down,
            feature=feature.detach().cpu().numpy())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_file', default=None, type=str, help='path to ply file')
    parser.add_argument(
        '--source_high_res',
        default=None,
        type=str,
        help='path to high_resolution point cloud')
    parser.add_argument(
        '--target', default=None, type=str, help='path to produce generated data')
    parser.add_argument(
        '-m',
        '--model',
        default=None,
        type=str,
        help='path to latest checkpoint (default: None)')
    parser.add_argument(
        '--voxel_size',
        default=0.05,
        type=float,
        help='voxel size to preprocess point cloud')
    parser.add_argument('--with_cuda', action='store_true')
    parser.add_argument(
        '--num_rand_keypoints',
        type=int,
        default=5000,
        help='Number of random keypoints for each scene')

    args = parser.parse_args()

    main(args)