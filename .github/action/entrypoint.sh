#!/bin/sh -l

cd /github/workspace
KDKNN_JAX_CUDA=yes python3 -m pip install .
python3 -c 'import kdknn_jax;print(kdknn_jax.__version__)'
python3 -c 'import kdknn_jax.gpu_ops'
