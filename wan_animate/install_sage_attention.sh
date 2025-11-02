#!/usr/bin/env bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
git reset --hard 68de379

export EXT_PARALLEL=4
export NVCC_APPEND_FLAGS="--threads 8"
export MAX_JOBS=32

/app/.venv/bin/python setup.py install