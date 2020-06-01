pythonPath=/vol/research/ucdatasets/MPSD/akin_tmp/miniconda3/envs/pytorch/bin/
sourcePath=/vol/research/ucdatasets/MPSD/akin_tmp/src/FCGF_submit/
dataDir=/vol/research/ucdatasets/MPSD/akin_tmp/src/FCGF/data/threedmatch/
outDir=/vol/research/ucdatasets/MPSD/akin_tmp/src/FCGF_submit/outputs

${pythonPath}/python ${sourcePath}/train.py --threed_match_dir ${dataDir} --out_dir ${outDir}
