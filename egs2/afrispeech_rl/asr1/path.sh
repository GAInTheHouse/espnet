MAIN_ROOT=$PWD/../../..

export PATH=$PWD/utils/:$MAIN_ROOT/tools/bin:$PATH
export LC_ALL=C

if [ -f "${MAIN_ROOT}/tools/activate_python.sh" ]; then
    . "${MAIN_ROOT}/tools/activate_python.sh"
fi

if [ -f "${MAIN_ROOT}/tools/extra_path.sh" ]; then
    . "${MAIN_ROOT}/tools/extra_path.sh"
fi

export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"
