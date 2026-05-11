#!/bin/bash
proj_name=DSRL_pi0_Libero
device_id=0
#测试1111
export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id
export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PYTHONPATH=$PWD:$PWD/LIBERO:$PWD/openpi/src:$PYTHONPATH
export LIBERO_CONFIG_PATH=$PWD/.libero
export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=$PWD/.cache/matplotlib
mkdir -p "$MPLCONFIGDIR"

pip install mujoco==3.3.1

python3 examples/launch_train_sim.py \
--algorithm pixel_sac \
--env libero \
--prefix dsrl_pi0_libero \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
--eval_interval 10000 \
--log_interval 500 \
--eval_episodes 10 \
--multi_grad_step 20 \
--start_online_updates 500 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 20 \
--hidden_dims 128 \
