#!/bin/bash

module load devel/miniforge/25.3.1-python-3.12
eval "$(conda shell.bash hook)"
#conda create -n xda python numpy scipy scikit-learn colorama
#conda activate xda
#conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
#pip install pyelftools
#conda deactivate

conda create -n xda python=3.7 numpy=1.21.5 scipy=1.7.3 scikit-learn=1.0.2 colorama=0.4.6 -y
conda activate xda
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pyelftools
cd "src/reveng_ml/ComparativeEvaluation/XDA"
pip install --editable .
conda deactivate
