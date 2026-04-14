# Command backend: "local" (single machine) or "slurm" for cluster.
cmd_backend='local'

if [ "${cmd_backend}" = local ]; then
    export train_cmd="run.pl"
    export cuda_cmd="run.pl"
    export decode_cmd="run.pl"
elif [ "${cmd_backend}" = slurm ]; then
    export train_cmd="slurm.pl"
    export cuda_cmd="slurm.pl"
    export decode_cmd="slurm.pl"
else
    echo "$0: Unknown cmd_backend=${cmd_backend}" 1>&2
    return 1
fi
