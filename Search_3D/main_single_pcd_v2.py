

import open3d as o3d
import numpy as np
import sys
import math
import os

import copy

## Visualization is taken from "https://github.com/chrischoy/FCGF
from util.visualization import get_colored_point_cloud_feature
from util.pointcloud import make_open3d_point_cloud

## VLAD library is from "https://github.com/jorjasso/VLAD"
from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import argparse
import glob
import cv2

import time

import random
from pathlib import Path


def points_2_pointcloud(coords):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    return pcd


def visualize_point_cloud(pcd_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    #ctr = vis.get_view_control()
    #print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    #ctr.change_field_of_view(step=fov_step)
    #print("Field of view (after changing) %.2f" % ctr.get_field_of_view())

    ## TODO:
    ## 1-> json cmaera parameters change H,W
    ## 2-> add screen capture feature

    vis.run()
    vis.destroy_window()

def convertMeshBox2LineBox(mesh_box, color_select):
    points = np.array(mesh_box.vertices)
    lines = [[0, 1], [0, 2], [1, 3], [2, 3],
             [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7], ]

    ##colors = [[1, 0, 0] for i in range(len(lines))]
    colors = [color_select for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

class Search3D:

    def __init__(self, path_to_pcd, path_to_feat, isVisualizationON):
        self.path_to_pcd = path_to_pcd # Path to point cloud file
        self.path_to_feat = path_to_feat # Path to feature file
        self.voxel_size = 0.025
        self.read_inputs()
        self.k = 16  # no. of visual words used for VisualDictionary Generation
        self.sample_step_size = 30 #100
        self.leafSize = 40 # leafsize for "indexBallTree"
        self.k_retrieve = 3 # number of retrieved box
        self.color_dict={"black":[0,0,0], "blue":[0,0,1]}
        self.isSearchAvaliable = True
        self.visualization = isVisualizationON

    def read_inputs(self):
        data_i = np.load(self.path_to_feat)
        self.coord_i, self.points_i, self.feat_i = data_i['xyz'], data_i['points'], data_i['feature']
        self.pcd_i = points_2_pointcloud(self.coord_i)

    def computeVisualDictionary(self):
        descriptors = self.feat_i
        self.visualDictionary = kMeansDictionary(descriptors, self.k)

    def extractBoxes_VLADdesc(self):

        self.descriptorsVLAD=list()
        self.idBox = list()
        self.descriptorFCGF=list()
        self.pointCoords=list()
        self.meshBox=list()

        ## For each box in the point cloud, VLAD descriptors are computed.
        for ind_p in list(range(0, self.coord_i.shape[0],self.sample_step_size)):

            ## Create mesh_box - experiment

            ## Box width, this value is computed considering the calibration of datasets in '3DMatch' repository
            box_w = 0.2

            ## Creation of a box
            mesh_box = o3d.geometry.TriangleMesh.create_box(width=box_w, height=box_w, depth=box_w)
            mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

            ## Locate center of box to the origin
            mat_trans = np.eye((4))
            mat_trans[0, 3] = -box_w/2
            mat_trans[1, 3] = -box_w/2
            mat_trans[2, 3] = -box_w/2
            mesh_box.transform(mat_trans)

            ## Locate center of box to the point location
            mat_trans = np.eye((4))
            mat_trans[0, 3] = self.coord_i[ind_p, 0]
            mat_trans[1, 3] = self.coord_i[ind_p, 1]
            mat_trans[2, 3] = self.coord_i[ind_p, 2]
            mesh_box.transform(mat_trans)

            ## We store the all boxes in a list named "self.meshBox"
            self.meshBox.append(mesh_box)

            ## Sampling Points in the Box:

            thresh = math.sqrt(3)*box_w/2
            q_point = self.coord_i[ind_p, :]
            q_point_arr = np.tile(q_point, (self.coord_i.shape[0], 1))
            dist_arr = q_point_arr - self.coord_i
            dist = np.linalg.norm(dist_arr, axis=1)
            box_p_ind = np.where(dist<=thresh)[0]

            ## Container for coordinates of points in the Box
            box_p = self.coord_i[box_p_ind, :]
            ## Container for FCGF features of points in the Box
            box_p_feat = self.feat_i[box_p_ind, :]

            ## Calling VLAD descriptor extractor
            if box_p_feat is not None:
                ## Previously computed "self.visualDictionary" is used here
                ## VLAD function is from VLAD library (https://github.com/jorjasso/VLAD)
                v = VLAD(box_p_feat, self.visualDictionary)
                self.descriptorsVLAD.append(v)
                self.idBox.append(ind_p)

                self.descriptorFCGF.append(box_p_feat)
                self.pointCoords.append(box_p)

        self.descriptorsVLAD = np.asarray(self.descriptorsVLAD)

        self.No_box = len(self.idBox)

    def computeIndexBallTree(self):
        self.tree = indexBallTree(self.descriptorsVLAD, self.leafSize)

    ##
    #  Inputs:
    #   boxId: Index of the Query Box
    #   k_NN:  k Nearest Neighbor
    ##
    def query(self, boxId, k_NN):

        self.k_retrieve = k_NN

        ## Initialization - Computation of Colored Point Cloud Based on FCGF Features
        ## Duplication of "pcd_i"point cloud
        ## We show matched boxes on this point cloud
        pcd_match = points_2_pointcloud(self.pcd_i.points)

        ## Translate pcd match to the right for visualization
        mat_trans = np.eye(4)
        mat_trans[0, 3] = 3.0  # 4.0
        mat_trans[1, 3] = 0
        mat_trans[2, 3] = 0
        pcd_match.transform(mat_trans)

        ## We used point cloud coloring based on FCGF features
        ## This coloring is also used in FCGF paper
        if self.visualization:
            spheres_i = get_colored_point_cloud_feature(self.pcd_i, self.feat_i, self.voxel_size)
            spheres_match_i = get_colored_point_cloud_feature(pcd_match, self.feat_i, self.voxel_size)
        else:
            spheres_i = self.pcd_i
            spheres_match_i = pcd_match

        ## TODO: interactive box searching
        ## How many boxes we have.

        while(self.isSearchAvaliable):

            ## Fetching the feature vector of the box, which is previously computed
            queryBox_descriptor_FGCF = self.descriptorFCGF[boxId]
            v = VLAD(queryBox_descriptor_FGCF, self.visualDictionary)
            v = v.reshape(1, -1)

            # find the k most relevant images
            # using previously generated "balltree"
            dist, ind = self.tree.query(v, self.k_retrieve)

            ## Initialization of Visuzation - Empty open3D Scene
            visual_list = []

            visual_list.append(spheres_i)
            visual_list.append(spheres_match_i)

            # Draw the box - Query
            mesh_box_vertices_query = self.meshBox[boxId]
            ## Matched box is colored in black
            lines_set_query_box = convertMeshBox2LineBox(mesh_box_vertices_query, self.color_dict["black"])
            visual_list.append(lines_set_query_box)


            ## Iteration through neaarest neighor matches
            ## and draw each box on the point cloud
            for ind_match in ind[0]:

                ## Draw the box - Match
                mesh_box_vertices_match = copy.deepcopy(self.meshBox[ind_match])
                mesh_box_vertices_match.transform(mat_trans)
                ## Matched box is colored in blue
                lines_set_match_box = convertMeshBox2LineBox(mesh_box_vertices_match, self.color_dict["blue"])

                visual_list.append(lines_set_match_box)


            if self.visualization:
                visualize_point_cloud(visual_list)

                decision = input('Do you want to continue to searching another box? Y or N? \n')

                if decision.capitalize() == 'Y':
                    selected_boxId = input('Select boxId for another query search between 0 and {} \n'.format(self.No_box))
                    boxId = int(selected_boxId)
                    print('Another query search is started using boxId = {} \n'.format(boxId))
                elif decision.capitalize() == 'N':
                    self.isSearchAvaliable = False
                else:
                    print('Another query search is started using boxId = {} \n'.format(boxId))
            else:
                self.isSearchAvaliable = False


def main(args):

    ## Reading the arguments
    args = vars(args)
    PATH_PCD = args["path_pointcloud"]
    PATH_FEATURE = args["path_feature"]
    k_NN = args["k_nearest_neighbor"]
    isVisualizationON = bool(args["visualization"])


    file_path_pcd_i = PATH_PCD
    file_path_feat_i = PATH_FEATURE

    ## TODO:



    ##

    if os.path.isfile(file_path_pcd_i)==0 or os.path.isfile(file_path_feat_i)==0:

        print('ERROR - Missing Files - Check Files ')
        print('Point cloud file = ', file_path_pcd_i)
        print('Feature file = ', file_path_feat_i)

        sys.exit()

    ## Start timer
    if not isVisualizationON:
        start_time = time.time()

    ###
    #  "Search_3D" Class Instance Generation
    ###

    s3d = Search3D(file_path_pcd_i, file_path_feat_i, isVisualizationON)

    ###
    # Visual Dictionary Generation
    ###
    print('Computing Visual Dictionary')
    s3d.computeVisualDictionary()

    ###
    # VLAD Decriptor Extractor
    ###
    print('Computing VLAD Descriptors')
    s3d.extractBoxes_VLADdesc()

    ###
    # IndexballTree Generation
    ###
    print('Generating of IndexBallTree')
    s3d.computeIndexBallTree()

    ###
    # Query Search
    ###
    print('Search Box Query in Point Cloud')
    boxId = 0
    s3d.query(boxId, k_NN)

    ## end timer
    if not isVisualizationON:
        execution_time = time.time() - start_time
        print('execution time = %.2f seconds' % execution_time)


if __name__ == "__main__":
    # execute only if run as a script

    parser = argparse.ArgumentParser(description='Search 3D Application')


    # Dataset setting
    parser.add_argument("-p", "--path_pointcloud", required = True,
                        help = "Path of the point cloud file")

    parser.add_argument("-f", "--path_feature", required = True,
                        help = "Path of the FCGF feature file associated the input point cloud")

    parser.add_argument("-k", "--k_nearest_neighbor", default=3, type=int, required=True,
                        help="k nearest neighbor matches are computed in the program")

    parser.add_argument("-v", "--visualization", default=1, type=int, required=True,
                        help="Visualization Flag")


    args = parser.parse_args()
    print('Input Arguments:')
    for arg in vars(args):
        print("\t {} -> {}".format(arg, getattr(args, arg)))


    print('Search_3D is running')

    main(parser.parse_args())
    ## TODO: add args.