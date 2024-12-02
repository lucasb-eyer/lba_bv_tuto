#!/usr/bin/env fish

# Either run this file in the big_vision folder with the venv active,
# or let's just hard-code this right here to avoid annoying mistakes:
cd /home/big_vision
. ../venv/bin/activate.fish

set -l COMMAND env \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  python -m big_vision.train

for lr in 1e-2 3e-3 1e-3 3e-4 1e-4
 for wd in 1e-1 1e-2 1e-3 1e-4 1e-5
  for ep in 10 30 100
   for bs in 8 16 32
    env TS_SOCKET=/tmp/ts_socket_$bs ts -S 9
    for aug in flip=false,crop=false flip=true,crop=true flip=true,crop=false flip=false,crop=true
     env TS_SOCKET=/tmp/ts_socket_$bs ts -G 1 $COMMAND --config /workspace/lba_bv_tuto/lines_resnet.py:$aug \
       --workdir /workspace/workdir_sweep_(date '+%m-%d_%H%M')_d26_lr{$lr}_wd{$wd}_ep{$ep}_bs{$bs}_{$aug} \
       --config.model.depth=26 \
       --config.input.batch_size=$bs \
       --config.total_epochs=$ep \
       --config.lr=$lr \
       --config.wd=$wd
    end
   end
  end
 end
end
