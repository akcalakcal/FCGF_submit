#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/akin/workspace/workspace_dependencies/conda/miniconda3 -s
#rm Miniconda3-latest-Linux-x86_64.sh
conda=/home/akin/workspace/workspace_dependencies/conda/miniconda3/bin/conda
$conda create -y -n pytorch python=3.7 pip
$conda install -n pytorch -y pytorch torchvision cudatoolkit=10.1 -c pytorch
