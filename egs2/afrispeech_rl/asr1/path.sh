MAIN_ROOT=$PWD/../../..

export PATH=$PWD/utils/:$MAIN_ROOT/tools/bin:$PATH
export LC_ALL=C

if [ -f "${MAIN_ROOT}/tools/activate_python.sh" ]; then
    . "${MAIN_ROOT}/tools/activate_python.sh"
fi

if [ -f "${MAIN_ROOT}/tools/extra_path.sh" ]; then
    . "${MAIN_ROOT}/tools/extra_path.sh"
fi

# If running inside a conda environment (e.g. espnet_rl), ensure that
# environment's python3 takes priority over /usr/bin/python3.
# run.pl spawns subshells that source path.sh; without this, macOS system
# Python is used and packages installed in the conda env are invisible.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

# Reduce CUDA allocator fragmentation (recommended by PyTorch for OOM errors)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
