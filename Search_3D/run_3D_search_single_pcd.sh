
pythonPath=/home/akincaliskan/workspace/workspace_dependicies/Anaconda/Anaconda/envs/Search_3d/bin/
sourcePath=./
pointCloudFile=../data_test/tmp/7-scenes-redkitchen/cloud_bin_0.ply
feature_FCGF_File=../features_tmp/7-scenes-redkitchen/cloud_bin_0.npz


${pythonPath}/python ${sourcePath}/main_single_pcd.py -p ${pointCloudFile} -f ${feature_FCGF_File} -k 3 -v 1



