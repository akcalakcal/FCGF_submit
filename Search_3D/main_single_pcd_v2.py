

import open3d as o3d
import numpy as np
import sys
import math
import os

import copy

import tkinter.filedialog


from concurrent.futures import ThreadPoolExecutor



from lib.feature_extractor import FeatureExtractor


## Visualization is taken from "https://github.com/chrischoy/FCGF
from utils.visualization import get_colored_point_cloud_feature
from utils.pointcloud import make_open3d_point_cloud

## VLAD library is from "https://github.com/jorjasso/VLAD"
from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import argparse
import glob
import cv2

import time

from tqdm import tqdm


import random
from pathlib import Path


def points_2_pointcloud(coords):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    #colors = [[0.5, 0.5, 0.5] for i in range(len(pcd.points))]
    #pcd.colors = o3d.utility.Vector3dVector(colors)
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

    #vis.get_render_option().load_from_json("./renderoption.json")

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

    def __init__(self, path_to_pcd, path_query_pcd, path_to_feat, isVisualizationON, input_type):
        self.path_to_pcd = path_to_pcd # Path to point cloud file
        self.path_to_feat = path_to_feat # Path to feature file
        self.path_query_pcd = path_query_pcd
        self.voxel_size = 0.025
        self.read_inputs()
        self.k = 4 #2 #16  # no. of visual words used for VisualDictionary Generation
        self.sample_step_size = 10 #100 #300 #30 #100
        self.leafSize = 40 # leafsize for "indexBallTree"
        self.k_retrieve = 3 # number of retrieved box
        self.color_dict={"black":[0,0,0], "blue":[0,0,1]}
        self.isSearchAvaliable = True
        self.visualization = isVisualizationON
        self.input_type = input_type
        self.pcd_apart = 10

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
        #for ind_p in list(range(0, 10000,self.sample_step_size)):

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

    ##
    # With given bounding box, search boxes are being modified
    ##
    def extractBoxes_VLADdesc_given_BB(self):

        self.descriptorsVLAD=list()
        self.idBox = list()
        self.descriptorFCGF=list()
        self.pointCoords=list()
        self.meshBox=list()

        ## DEBUG

        if self.input_type == 'mesh':
            pcd_in = o3d.io.read_triangle_mesh(self.path_query_pcd)
            pcd_in.compute_vertex_normals()
        if self.input_type == 'pcd':
            pcd_in = o3d.io.read_point_cloud(self.path_query_pcd)
        dummy_box = pcd_in.get_axis_aligned_bounding_box()

        box_scale = 1.2  # 0.5
        box_w_max = dummy_box.max_bound
        box_w_min = dummy_box.min_bound
        box_w_x = box_scale * abs(box_w_max[0] - box_w_min[0])
        box_w_y = box_scale * abs(box_w_max[1] - box_w_min[1])
        box_w_z = box_scale * abs(box_w_max[2] - box_w_min[2])


        ## DEBUG

        ## For each box in the point cloud, VLAD descriptors are computed.
        ##for ind_p in list(range(0, self.coord_i.shape[0],self.sample_step_size)):
        for ind_p in tqdm(range(0, self.coord_i.shape[0],self.sample_step_size)):
        #for ind_p in list(range(0, 10000,self.sample_step_size)):

            ## Create mesh_box - experiment

            ## Box width, this value is computed considering the calibration of datasets in '3DMatch' repository
            #box_w = 0.2

            ## DEBUG

            ## Creation of a box
            mesh_box = o3d.geometry.TriangleMesh.create_box(width=box_w_x, height=box_w_y, depth=box_w_z)
            mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

            ## Locate center of box to the origin
            mat_trans = np.eye((4))
            mat_trans[0, 3] = -box_w_x / 2
            mat_trans[1, 3] = -box_w_y / 2
            mat_trans[2, 3] = -box_w_z / 2
            mesh_box.transform(mat_trans)

            ## DEBUG

            '''
            ## Creation of a box
            mesh_box = o3d.geometry.TriangleMesh.create_box(width=box_w, height=box_w, depth=box_w)
            mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

            ## Locate center of box to the origin
            mat_trans = np.eye((4))
            mat_trans[0, 3] = -box_w/2
            mat_trans[1, 3] = -box_w/2
            mat_trans[2, 3] = -box_w/2
            mesh_box.transform(mat_trans)
            
            '''

            ## Locate center of box to the point location
            mat_trans = np.eye((4))
            mat_trans[0, 3] = self.coord_i[ind_p, 0]
            mat_trans[1, 3] = self.coord_i[ind_p, 1]
            mat_trans[2, 3] = self.coord_i[ind_p, 2]
            mesh_box.transform(mat_trans)

            ## We store the all boxes in a list named "self.meshBox"
            self.meshBox.append(mesh_box)

            ## Sampling Points in the Box:

            box_w = max(box_w_x, box_w_y, box_w_z)

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
                #self.idBox.append(ind_p)

                #self.descriptorFCGF.append(box_p_feat)
                #self.pointCoords.append(box_p)

        self.descriptorsVLAD = np.asarray(self.descriptorsVLAD)

        #self.No_box = len(self.idBox)

    ##
    # With multi thread
    ##

    def extractBoxes_VLADdesc_given_BB_multhread(self):

        self.descriptorsVLAD = list()

        if self.input_type == 'mesh':
            pcd_in = o3d.io.read_triangle_mesh(self.path_query_pcd)
            pcd_in.compute_vertex_normals()
        if self.input_type == 'pcd':
            pcd_in = o3d.io.read_point_cloud(self.path_query_pcd)
        dummy_box = pcd_in.get_axis_aligned_bounding_box()

        box_scale = 1.2  # 0.5
        box_w_max = dummy_box.max_bound
        box_w_min = dummy_box.min_bound
        box_w_x = box_scale * abs(box_w_max[0] - box_w_min[0])
        box_w_y = box_scale * abs(box_w_max[1] - box_w_min[1])
        box_w_z = box_scale * abs(box_w_max[2] - box_w_min[2])

        ## Multi thread - debug

        ## TODO: I am here.

        ## Multi thread - debug

        ## For each box in the point cloud, VLAD descriptors are computed.
        ##for ind_p in list(range(0, self.coord_i.shape[0],self.sample_step_size)):
        for ind_p in tqdm(range(0, self.coord_i.shape[0], self.sample_step_size)):


            ## Creation of a box
            mesh_box = o3d.geometry.TriangleMesh.create_box(width=box_w_x, height=box_w_y, depth=box_w_z)
            mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

            ## Locate center of box to the origin
            mat_trans = np.eye((4))
            mat_trans[0, 3] = -box_w_x / 2
            mat_trans[1, 3] = -box_w_y / 2
            mat_trans[2, 3] = -box_w_z / 2
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

            box_w = max(box_w_x, box_w_y, box_w_z)

            thresh = math.sqrt(3) * box_w / 2
            q_point = self.coord_i[ind_p, :]
            q_point_arr = np.tile(q_point, (self.coord_i.shape[0], 1))
            dist_arr = q_point_arr - self.coord_i
            dist = np.linalg.norm(dist_arr, axis=1)
            box_p_ind = np.where(dist <= thresh)[0]

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


        self.descriptorsVLAD = np.asarray(self.descriptorsVLAD)


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

        #pcd_match = points_2_pointcloud(self.pcd_i.points)

        ## DEBUG read mesh instead of point cloud
        if self.input_type == 'mesh':
            self.pcd_i = o3d.io.read_triangle_mesh(self.path_to_pcd)
            self.pcd_i.compute_vertex_normals()
            pcd_match = o3d.io.read_triangle_mesh(self.path_to_pcd)
            pcd_match.compute_vertex_normals()
        elif self.input_type == 'pcd':
            self.pcd_i = o3d.io.read_point_cloud(self.path_to_pcd)
            pcd_match = o3d.io.read_point_cloud(self.path_to_pcd)

        #o3d.visualization.draw_geometries([pcd_match])
        ## DEBUG

        ## Translate pcd match to the right for visualization
        mat_trans = np.eye(4)
        mat_trans[0, 3] = 15.0 #3.0  # 4.0
        mat_trans[1, 3] = 0
        mat_trans[2, 3] = 0
        pcd_match.transform(mat_trans)

        ## We used point cloud coloring based on FCGF features
        ## This coloring is also used in FCGF paper
        if self.visualization:
            #spheres_i = get_colored_point_cloud_feature(self.pcd_i, self.feat_i, self.voxel_size)
            #spheres_match_i = get_colored_point_cloud_feature(pcd_match, self.feat_i, self.voxel_size)
            spheres_i = self.pcd_i
            spheres_match_i = pcd_match
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

    def query_given_BB(self, boxId, k_NN, feat_extractor):

        self.k_retrieve = k_NN

        ## Initialization - Computation of Colored Point Cloud Based on FCGF Features
        ## Duplication of "pcd_i"point cloud
        ## We show matched boxes on this point cloud

        #pcd_match = points_2_pointcloud(self.pcd_i.points)

        ## DEBUG read mesh instead of point cloud
        if self.input_type == 'mesh':
            self.pcd_i = o3d.io.read_triangle_mesh(self.path_to_pcd)
            self.pcd_i.compute_vertex_normals()
            pcd_match = o3d.io.read_triangle_mesh(self.path_to_pcd)
            pcd_match.compute_vertex_normals()
        elif self.input_type == 'pcd':
            self.pcd_i = o3d.io.read_point_cloud(self.path_to_pcd)
            pcd_match = o3d.io.read_point_cloud(self.path_to_pcd)
        #o3d.visualization.draw_geometries([pcd_match])

        ## Distance between point clouds

        dummy_box = pcd_match.get_axis_aligned_bounding_box()

        box_w_max = dummy_box.max_bound
        box_w_min = dummy_box.min_bound
        self.pcd_apart = 1.5 * abs(box_w_max[0] - box_w_min[0])


        ## Distance between point clouds


        ## DEBUG

        ## Translate pcd match to the right for visualization
        mat_trans = np.eye(4)
        mat_trans[0, 3] = self.pcd_apart #3.5 #3.0  # 4.0
        mat_trans[1, 3] = 0
        mat_trans[2, 3] = 0
        pcd_match.transform(mat_trans)

        ## We used point cloud coloring based on FCGF features
        ## This coloring is also used in FCGF paper
        if self.visualization:
            #spheres_i = get_colored_point_cloud_feature(self.pcd_i, self.feat_i, self.voxel_size)
            #spheres_match_i = get_colored_point_cloud_feature(pcd_match, self.feat_i, self.voxel_size)
            spheres_i = self.pcd_i
            spheres_match_i = pcd_match
        else:
            spheres_i = self.pcd_i
            spheres_match_i = pcd_match

        ## TODO: interactive box searching
        ## How many boxes we have.

        while(self.isSearchAvaliable):

            ## TODO: extract FCGF feature for query point cloud here

            #target_folder_path = os.path.dirname(os.path.abspath(self.path_query_pcd))
            #file_path_query_feat_i = feat_extractor.extract(self.path_query_pcd, target_folder_path)

            #data_query_i = np.load(file_path_query_feat_i)
            #query_coord_i, query_points_i, query_feat_i = data_query_i['xyz'], data_query_i['points'], data_query_i['feature']
            #query_pcd_i = points_2_pointcloud(query_coord_i)

            ## DEBUG

            pcd_in_query = o3d.io.read_triangle_mesh(self.path_query_pcd)
            BB_pcd_in_query = pcd_in_query.get_axis_aligned_bounding_box()


            box_p_query_ind = []
            for p_q in pcd_in_query.vertices:
                index_pos = np.where((self.coord_i[:,0] == p_q[0]) & (self.coord_i[:, 1] == p_q[1]) & (self.coord_i[:, 2] == p_q[2]))
                if index_pos[0]:
                    box_p_query_ind.append(index_pos[0])

            ## Container for FCGF features of points in the query Box
            box_p_query_ind = np.array(box_p_query_ind)[:,0]
            box_p_query_feat = self.feat_i[box_p_query_ind, :]


            ## DEBUG

            #

            ## Fetching the feature vector of the box, which is previously computed
            queryBox_descriptor_FGCF = box_p_query_feat #self.descriptorFCGF[boxId]
            #queryBox_descriptor_FGCF = query_feat_i
            v = VLAD(queryBox_descriptor_FGCF, self.visualDictionary)
            v = v.reshape(1, -1)

            ## DEBUG - kretrieve
            search_continue = True

            while search_continue:

                # find the k most relevant images
                # using previously generated "balltree"
                dist, ind = self.tree.query(v, self.k_retrieve)

                ## Initialization of Visuzation - Empty open3D Scene
                visual_list = []

                visual_list.append(spheres_i)
                visual_list.append(spheres_match_i)

                # Draw the box - Query


                visual_list.append(BB_pcd_in_query)


                ## Iteration through neaarest neighor matches
                ## and draw each box on the point cloud
                mesh_box_stack = []
                tmp_cnt = 0
                for ind_match in ind[0]:

                    # Init
                    IoU = False

                    ## Draw the box - Match
                    mesh_box_vertices_match = copy.deepcopy(self.meshBox[ind_match])

                    if tmp_cnt == 0:
                        mesh_box_stack.append(copy.deepcopy(mesh_box_vertices_match))
                        mesh_box_vertices_match.transform(mat_trans)
                        ## Matched box is colored in blue
                        lines_set_match_box = convertMeshBox2LineBox(mesh_box_vertices_match, self.color_dict["blue"])
                        visual_list.append(lines_set_match_box)

                    ## TODO: Compare matched mesh boxes wrt Intersection over Union (IoU)
                    if tmp_cnt > 0:

                        #IoU = mesh_box_stack[-1].is_intersecting(mesh_box_vertices_match)

                        for m_tmp in mesh_box_stack:
                            IoU_t = m_tmp.is_intersecting(mesh_box_vertices_match)
                            IoU = IoU or IoU_t

                        if not IoU:
                            mesh_box_stack.append(copy.deepcopy(mesh_box_vertices_match))

                            mesh_box_vertices_match.transform(mat_trans)
                            ## Matched box is colored in blue
                            lines_set_match_box = convertMeshBox2LineBox(mesh_box_vertices_match, self.color_dict["blue"])

                            visual_list.append(lines_set_match_box)


                    #visual_list_tmp = visual_list.copy()
                    #visual_list_tmp.append(lines_set_match_box)

                    #visualize_point_cloud(visual_list_tmp)

                    tmp_cnt = tmp_cnt + 1

                #print('len(visual_list) = ', len(visual_list))
                #print('self.k_retrieve = ', self.k_retrieve)
                #print('k_NN = ', k_NN)
                if len(visual_list) >= (k_NN + 2):
                    search_continue = False
                else:
                    self.k_retrieve = self.k_retrieve + 10

            ## DEBUG - kretrieve

            if self.visualization:
                visualize_point_cloud(visual_list)

                decision = input('Do you want to continue to searching another box? Y or N? \n')

                if decision.capitalize() == 'Y':
                    #selected_boxId = input('Select boxId for another query search between 0 and {} \n'.format(self.No_box))
                    #boxId = int(selected_boxId)
                    ## Select A Bounding Box Again
                    if self.input_type == 'pcd':
                        # pcd_in = o3d.io.read_triangle_mesh(file_path_pcd_i)
                        pcd_in = o3d.io.read_point_cloud(self.path_to_pcd)
                        # pcd_in.compute_vertex_normals()
                        # o3d.visualization.draw_geometries([pcd_in])
                        demo_crop_geometry(pcd_in)
                    elif self.input_type == 'mesh':
                        pcd_in = o3d.io.read_triangle_mesh(self.path_to_pcd)
                        pcd_in.compute_vertex_normals()
                        demo_crop_geometry(pcd_in)

                    ## Extract Vlad Descriptors given new =ly selected BB
                    self.extractBoxes_VLADdesc_given_BB()

                    self.query_given_BB(boxId, k_NN, feat_extractor)
                    print('Another query search is started using boxId = {} \n'.format(boxId))
                elif decision.capitalize() == 'N':
                    self.isSearchAvaliable = False
                else:
                    print('Another query search is started using boxId = {} \n'.format(boxId))
            else:
                self.isSearchAvaliable = False



def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def demo_crop_geometry(pcd):
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    #pcd = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    o3d.visualization.draw_geometries_with_editing([pcd])

def main(args):

    ## Reading the arguments
    args = vars(args)
    #PATH_PCD = args["path_pointcloud"]

    #PATH_PCD = "/home/akin/workspace/All_Data/Indoor_Lidar_RGBD_Scan_Dataset/Apartment/Reconstruction/ours_apartment/apartment.ply"
    #PATH_PCD = "/home/akin/workspace/workspace_applications/Deep_3D_Search/FCGF_submit/Search_3D/query_pcd/cropped_7_frag.ply"
    #PATH_QUERY_PCD = "/home/akin/workspace/workspace_applications/Deep_3D_Search/FCGF_submit/Search_3D/query_pcd/cropped_7.ply"

    #PATH_PCD = "/home/akin/workspace/All_Data/Tanks_and_Templates/Caterpillar/GT/Caterpillar.ply"
    PATH_PCD = "/home/akin/workspace/workspace_applications/Deep_3D_Search/FCGF_submit/Search_3D/query_pcd/cropped_1.ply"
    PATH_QUERY_PCD = "/home/akin/workspace/workspace_applications/Deep_3D_Search/FCGF_submit/Search_3D/query_pcd/cropped_query.ply"

    ## User dialog for input file


    PATH_PCD = tkinter.filedialog.askopenfilename()


    ##

    input_type = 'pcd' #'mesh'
    selection_tool = 1
    FCGF_vis = 0 #False

    PATH_FEATURE = args["path_feature"]
    k_NN = args["k_nearest_neighbor"]
    isVisualizationON = bool(args["visualization"])


    file_path_pcd_i = PATH_PCD
    file_path_query_pcd = PATH_QUERY_PCD
    file_path_feat_i = PATH_FEATURE



    ## TODO: EXP: Bounding box of the 3D geometry

    '''

    pcd_in = o3d.io.read_triangle_mesh(file_path_query_pcd)
    pcd_in.compute_vertex_normals()
    dummy_box = pcd_in.get_axis_aligned_bounding_box()
    #o3d.visualization.draw_geometries([dummy_box, pcd_in])

    
    ## DEBUG

    center_bb = dummy_box.get_center()

    box_w = 0.9
    box_scale = 1.2 #0.5
    box_w_max = dummy_box.max_bound
    box_w_min = dummy_box.min_bound
    box_w_x = box_scale*abs(box_w_max[0] - box_w_min[0])
    box_w_y = box_scale*abs(box_w_max[1] - box_w_min[1])
    box_w_z = box_scale*abs(box_w_max[2] - box_w_min[2])
    ## Creation of a box
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=box_w_x, height=box_w_y, depth=box_w_z)
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

    ## Locate center of box to the origin
    mat_trans = np.eye((4))
    mat_trans[0, 3] = -box_w_x / 2
    mat_trans[1, 3] = -box_w_y / 2
    mat_trans[2, 3] = -box_w_z / 2
    mesh_box.transform(mat_trans)

    ## Locate center of box to the point location
    mat_trans = np.eye((4))
    mat_trans[0, 3] = center_bb[0]
    mat_trans[1, 3] = center_bb[1]
    mat_trans[2, 3] = center_bb[2]
    mesh_box.transform(mat_trans)

    line_set_dummy = convertMeshBox2LineBox(mesh_box, [1,0,0])

    o3d.visualization.draw_geometries([dummy_box, pcd_in, line_set_dummy])

    ## DEBUG
    '''

    ##

    ## TODO: Add volume selection tool for the user here

    if selection_tool:
        if input_type == 'pcd':
            #pcd_in = o3d.io.read_triangle_mesh(file_path_pcd_i)
            pcd_in = o3d.io.read_point_cloud(file_path_pcd_i)
            #pcd_in.compute_vertex_normals()
            #o3d.visualization.draw_geometries([pcd_in])
            demo_crop_geometry(pcd_in)
        elif input_type == 'mesh':
            pcd_in = o3d.io.read_triangle_mesh(file_path_pcd_i)
            pcd_in.compute_vertex_normals()
            demo_crop_geometry(pcd_in)
        #sys.exit()
    ##

    ## TODO: Add feature extraction module here

    model_path = "/home/akin/workspace/workspace_applications/Deep_3D_Search/FCGF_submit/outputs_trained_models/checkpoint.pth"

    feat_extractor = FeatureExtractor(model_path)
    target_folder_path = os.path.dirname(os.path.abspath(file_path_pcd_i))
    file_path_feat_i = feat_extractor.extract(PATH_PCD, target_folder_path)

    ##



    ## TODO: Visualize input point cloud FCGF features
    if FCGF_vis:
        data_i = np.load(file_path_feat_i)
        coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
        pcd_i = o3d.io.read_point_cloud(file_path_pcd_i)
        pcd_match = points_2_pointcloud(coord_i)
        voxel_size = 0.05 #0.025

        pcd_in_FCGF = get_colored_point_cloud_feature(pcd_match, feat_i, voxel_size)

        file_name_FCGF_folder = os.path.dirname(os.path.abspath(file_path_query_pcd))
        file_name_FCGF_name = os.path.basename(file_path_query_pcd)
        file_name_FCGF_name = os.path.splitext(file_name_FCGF_name)[0]
        file_name_FCGF = file_name_FCGF_folder + "/" + file_name_FCGF_name + "_FCGF_" + str(voxel_size) + ".ply"
        o3d.io.write_triangle_mesh(file_name_FCGF, pcd_in_FCGF)
        o3d.visualization.draw_geometries([pcd_in_FCGF])

        sys.exit()


    '''
    ## TODO: Visualize input point cloud FCGF features - Query

    feat_extractor = FeatureExtractor(model_path)
    target_query_folder_path = os.path.dirname(os.path.abspath(file_path_query_pcd))
    file_path_query_feat_i = feat_extractor.extract(file_path_query_pcd, target_query_folder_path)

    data_i = np.load(file_path_query_feat_i)
    coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
    pcd_i = o3d.io.read_point_cloud(file_path_query_pcd)
    pcd_match = points_2_pointcloud(coord_i)
    voxel_size = 0.025
    pcd_in_FCGF = get_colored_point_cloud_feature(pcd_match, feat_i, voxel_size)

    o3d.visualization.draw_geometries([pcd_in_FCGF])

    sys.exit()

    #
    '''


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

    s3d = Search3D(file_path_pcd_i, file_path_query_pcd, file_path_feat_i, isVisualizationON, input_type)

    ###
    # Visual Dictionary Generation
    ###
    print('Computing Visual Dictionary')
    s3d.computeVisualDictionary()

    ###
    # VLAD Decriptor Extractor
    ###
    print('Computing VLAD Descriptors')
    #s3d.extractBoxes_VLADdesc()
    #s3d.extractBoxes_VLADdesc_given_BB()
    s3d.extractBoxes_VLADdesc_given_BB_multhread()

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
    #s3d.query(boxId, k_NN)
    s3d.query_given_BB(boxId, k_NN, feat_extractor)

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