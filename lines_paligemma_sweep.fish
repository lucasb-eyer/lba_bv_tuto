#!/usr/bin/env fish

set -l COMMAND env \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  XLA_PYTHON_CLIENT_MEM_FRACTION=.99 \
  BV_GEMMA_DIR=/home/ \
  python -m big_vision.trainers.proj.paligemma.train

for ex in 8 32 128 512 1024
 for lr in 3e-5 1e-5 3e-6 1e-6
  for ep in 3 10 30 100 300
    ts -G 1 $COMMAND --config /workspace/big_vision_lines.py \
      --workdir /workspace/workdir_ex{$ex}_lr{$lr}_ep{$ep}_(date '+%m-%d_%H%M') \
      --config.input.data.stop=$ex \
      --config.input.batch_size=32 \
      --config.total_epochs=$ep \
      --config.lr=$lr
  end
 end
end
