pythonPath=/vol/research/ucdatasets/MPSD/akin_tmp/miniconda3/envs/pytorch/bin/

mainFolder=/vol/research/ucdatasets/MPSD/akin_tmp/src/

sourcePath=${mainFolder}/FCGF_submit/
dataFolder=${mainFolder}/FCGF_submit/data_test/tmp/
featureExtFolder=${mainFolder}/FCGF_submit/features_tmp/
trainedModel=${mainFolder}/outputs_trained_models/checkpoint.pth

${pythonPath}/python ${sourcePath}/scripts/single_pcd_feature_extraction.py --source_file ${dataFolder}/7-scenes-redkitchen/cloud_bin_0.ply --target ${featureExtFolder}/7-scenes-redkitchen/ --voxel_size 0.025 --model ${trainedModel} --with_cuda
