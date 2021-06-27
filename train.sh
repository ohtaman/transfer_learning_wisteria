#!/bin/bash

#PJM -L rscgrp=tutorial2-a
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -g gt00
#PJM -j


module load cuda/11.2
module load cudnn/8.1.0
module load nccl/2.9.6

source .venv/bin/activate

nvidia-smi

HOME=$PWD python -m transfer.train --batch_size=2048
