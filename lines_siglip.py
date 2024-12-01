# pylint: disable=line-too-long
# -*- mode: python; tab-width: 2; indent-tabs-mode: nil -*- vim: set tabstop=2 shiftwidth=2 expandtab:
r"""Finetunine SigLIP encoder classifier on the line intersection JSON-L dataset.

Command to run this config:

python -m big_vision.train \
    --config lines_siglip.py \
    --workdir workdirs/`date '+%m-%d_%H%M'`
"""

import big_vision.configs.common as bvcc


def training_data():
  """Creates training data config."""
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
    name='bv:jsonl',
    fname='/workspace/lines_data/train.jsonl',
    # The strings in the `image` key of the JSON-L are files in that folder:
    fopen_keys={'image': '/workspace/lines_data/'},
    # Or they could be URLs to download:
    # download_keys=['image'],
    stop=128,  # Only use 128 examples. float("inf") for all.
  )
  c.pp = '|'.join([
    # Even though the images are already 224, we'll still reshape them
    # in order to give the variable a static shape.
    'decode|resize(224)|value_range(-1, 1)',
    'onehot(3, key="label", key_result="labels")',
    'keep("image", "labels")',
  ])
  # Keep the whole dataset in RAM after first pass. Useful optimization for
  # small/mid-size datasets, but risks a host OOM for large datasets.
  c.cache_raw = True
  c.shuffle_buffer_size = 1000
  return c


def get_config():
  """Config for training."""
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.input = training_data()
  c.num_classes = 3
  c.log_training_steps = 2  # Short, so log frequently!

  # Instead of epochs, you can also use `total_examples` or `total_steps`.
  c.total_epochs = 15
  c.input.batch_size = 32
  c.optax_name = 'big_vision.scale_by_adafactor'
  c.lr = 1e-5
  c.schedule = dict(decay_type='cosine', warmup_percent=0.1)
  c.grad_clip_norm = 1.0
  c.loss = 'softmax_xent'

  # Model section.
  c.model_name = 'vit'
  c.model = dict(variant='So400m/14', pool_type='map', head_zeroinit=True, scan=True)
  # Our model is a plain ViT, but the SigLIP checkpoints contain both a ViT
  # and a text encoder in subkeys as {"img": THE_VIT, "txt": THE_TXT_ENC}.
  # Here the special syntax `:img` allows us to load only the "img" sub-tree:
  c.model_init = '/workspace/webli_en_so400m_224_57633886.npz:img'
  # When a checkpoint is loaded, big_vision complains if any param in the checkpoint
  # was not used, or if any param in the model was not loaded.
  # Here, we do want the fresh zero-init'ed head params, so we need to =
  # explicitly tell big_vision to not load them:
  c.model_load = dict(dont_load=['head/kernel', 'head/bias'])

  # FSDP strategy.
  c.mesh = [('data', -1)]
  c.sharding_strategy = [('.*', 'fsdp(axis="data")')]
  c.sharding_rules = [('act_batch', ('data',))]

  c.evals = {}
  c.evals['eval/acc'] = dict(
    type='classification',
    loss_name='softmax_xent',
    data=dict(
      name='bv:jsonl',
      fname='/workspace/lines_data/val.jsonl',
      fopen_keys={'image': '/workspace/lines_data/'},
    ),
    log_percent=1/8, skip_first=True, cache='final_data',
    pp_fn=c.input.pp,  # eval and train pp are same here.
  )
  # TODO: Maybe attach fewshot linear probe evaluator just to show?

  c.seed = 0
  return c
