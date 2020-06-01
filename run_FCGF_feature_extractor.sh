pythonPath=/vol/research/ucdatasets/MPSD/akin_tmp/miniconda3/envs/pytorch/bin/
sourcePath=/vol/research/ucdatasets/MPSD/akin_tmp/src/FCGF_submit/
dataFolder=/vol/research/ucdatasets/MPSD/akin_tmp/src/FCGF/data_test/tmp/
featureExtFolder=/vol/research/ucdatasets/MPSD/akin_tmp/src/FCGF_submit/features_tmp/
trainedModel=/vol/research/ucdatasets/MPSD/akin_tmp/src/FCGF_submit/outputs_trained_models/checkpoint.pth

${pythonPath}/python ${sourcePath}/scripts/benchmark_3dmatch.py --source ${dataFolder} --target ${featureExtFolder} --voxel_size 0.025 --model ${trainedModel} --extract_features --evaluate_feature_match_recall --with_cuda
