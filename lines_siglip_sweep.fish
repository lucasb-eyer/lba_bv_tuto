#!/usr/bin/env fish

# Either run this file in the big_vision folder with the venv active,
# or let's just hard-code this right here to avoid annoying mistakes:
cd /home/big_vision
. ../venv/bin/activate.fish

set -l COMMAND env \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  XLA_PYTHON_CLIENT_MEM_FRACTION=.99 \
  python -m big_vision.train

for lr in 3e-4 1e-4 3e-5 1e-5 3e-6 1e-6
  for ep in 3 10 30 100
   for bs in 8 16 32
    ts -G 1 $COMMAND --config /workspace/lba_bv_tuto/lines_siglip.py \
      --workdir /workspace/workdir_sweep_lr{$lr}_ep{$ep}_bs{$bs}_(date '+%m-%d_%H%M') \
      --config.input.batch_size=$bs \
      --config.total_epochs=$ep \
      --config.lr=$lr
    end
  end
end
