# Build the docker

To build it without any paligemma checkpoint baked in:

```
docker build --build-arg PRELOAD_PG= . -t lucasbeyer/big_vision_jax_runpod:cuda12_cp311
```

To build it with 224 and 448 pt paligemma checkpoints baked in:

```
env HF_TOKEN=PASTE_YOUR_TOKEN_HERE docker build --secret id=HF_TOKEN --build-arg PRELOAD_PG=224,448 . -t lucasbeyer/big_vision_jax_runpod:cuda12_cp311_pg224_pg448
```

# Run paligemma fine-tuning inside the docker

Copy your config file and data over into the container, for example into the `/workspace` folder.

```
. /home/venv/bin/activate
cd /home/big_vision
XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_MEM_FRACTION=.99 BV_GEMMA_DIR=/home/ \
  python -m big_vision.trainers.proj.paligemma.train \
  --config /workspace/config.py \
  --workdir /workspace/workdir_(date '+%m-%d_%H%M')
```

While developing, it can be convenient to auto-jump into a debugger on any error. For that, change the command to:

```
python -m pdb -c c -m big_vision.trainers...
```

Or with an existing example config file:

```
  --config big_vision/configs/proj/paligemma/transfers/infovqa.py:freeze_vit=true \
```

And append individual config changes like so:

```
--config.input.data.stop=100 --config.input.batch_size=32 --config.total_epochs=200
```

# Sweeping hyper-parmeters

TODO, document using `ts` and a sweep script.
