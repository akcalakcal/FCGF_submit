pythonPath=/vol/research/ucdatasets/MPSD/akin_tmp/miniconda3/envs/pytorch/bin/
sourcePath=/vol/research/ucdatasets/MPSD/akin_tmp/src/FCGF_submit/
dataFolder=/vol/research/ucdatasets/MPSD/akin_tmp/src/FCGF/data_test/tmp/
featureExtFolder=/vol/research/ucdatasets/MPSD/akin_tmp/src/FCGF_submit/features_tmp/
trainedModel=/vol/research/ucdatasets/MPSD/akin_tmp/src/outputs_trained_models/checkpoint.pth

${pythonPath}/python ${sourcePath}/scripts/single_pcd_feature_extraction.py --source_file ${dataFolder}/7-scenes-redkitchen/cloud_bin_0.ply --target ${featureExtFolder}/7-scenes-redkitchen/ --voxel_size 0.025 --model ${trainedModel} --with_cuda
