#!/bin/bash

module load devel/miniforge/25.3.1-python-3.12
cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" && conda activate xda
rc1=$?
if [ $rc1 -ne 0 ]; then
	echo "Failed to activate conda environment"
	exit $rc1
fi
python3 InferXDA.py $1 $2 &> InferXDA.log
rcInfer=$?
conda deactivate
exit $rcInfer
