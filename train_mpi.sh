#!/bin/bash

#PJM -L rscgrp=tutorial1-a
#PJM -L node=4
#PJM --mpi proc=4
#PJM -L elapse=1:00:00
#PJM -g gt00
#PJM -j

module load cuda/11.2
module load cudnn/8.1.0
module load gcc/8.3.1
module load nccl/2.9.6

module load ompi-cuda/4.1.1-11.2



source .venv/bin/activate
mpirun \
  -machinefile ${PJM_O_NODEINF} \
  -npernode 1 -x HOME=$PWD \
  python -m transfer.train \
    --mpi \
    --max_epochs=5 \
    --batch_size=2048 \
    --learning_rate=4e-3 \
    --model_dir=model.$PJM_SUBJOBID
