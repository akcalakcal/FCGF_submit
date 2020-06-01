pythonPath=/vol/research/ucdatasets/MPSD/akin_tmp/miniconda3/envs/pytorch/bin/

mainFolder=/vol/research/ucdatasets/MPSD/akin_tmp/src/

sourcePath=${mainFolder}/FCGF_submit/
dataDir=${mainFolder}/FCGF_submit/threedmatch/
outDir=${mainFolder}/FCGF_submit/outputs

${pythonPath}/python ${sourcePath}/train.py --threed_match_dir ${dataDir} --out_dir ${outDir}
