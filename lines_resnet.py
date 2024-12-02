# pylint: disable=line-too-long
# -*- mode: python; tab-width: 2; indent-tabs-mode: nil -*- vim: set tabstop=2 shiftwidth=2 expandtab:
r"""Train a small ResNet classifier from scratch on the line intersection JSON-L dataset.

Command to run this config:

python -m big_vision.train \
    --config lines_resnet.py \
    --workdir workdirs/`date '+%m-%d_%H%M'`
"""

import big_vision.configs.common as bvcc


def register_new_pp_ops():
  from big_vision.pp import utils
  from big_vision.pp.registry import Registry
  import tensorflow as tf

  @Registry.register("preprocess_ops.flip_ud")
  @utils.InKeyOutKey()
  def get_random_flip_ud():
    def _random_flip_ud_pp(image):
      return tf.image.random_flip_up_down(image)
    return _random_flip_ud_pp


def training_data(flip=True, crop=True):
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
    'flip_lr|flip_ud' if flip else '',
    'pad_to_shape(shape=(256, 256, 3), pad_value=1, where="both")' if crop else '',
    'random_crop(224)' if crop else '',
    'onehot(3, key="label", key_result="labels")',
    'keep("image", "labels")',
  ])
  # Keep the whole dataset in RAM after first pass. Useful optimization for
  # small/mid-size datasets, but risks a host OOM for large datasets.
  c.cache_raw = True
  c.shuffle_buffer_size = 1000
  return c


def get_config(arg=None):
  """Config for training."""
  register_new_pp_ops()
  c = bvcc.parse_arg(arg, flip=True, crop=True)
  c.input = training_data(c.flip, c.crop)
  c.num_classes = 3
  c.log_training_steps = 2  # Short, so log frequently!

  # Instead of epochs, you can also use `total_examples` or `total_steps`.
  c.total_epochs = 30
  c.input.batch_size = 8
  c.optax_name = 'big_vision.scale_by_adafactor'
  c.lr = 1e-4
  c.wd = 1e-2
  c.schedule = dict(decay_type='cosine', warmup_percent=0.1)
  c.grad_clip_norm = 1.0
  c.loss = 'softmax_xent'

  # Model section.
  c.model_name = 'bit'
  c.model = dict(depth=50, width=1.0)

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
    pp_fn='|'.join([
      'decode|resize(224)|value_range(-1, 1)',
      'onehot(3, key="label", key_result="labels")',
      'keep("image", "labels")',
    ])
  )
  c.evals['test/acc'] = dict(
    type='classification',
    loss_name='softmax_xent',
    data=dict(
      name='bv:jsonl',
      fname='/workspace/lines_data/test.jsonl',
      fopen_keys={'image': '/workspace/lines_data/'},
    ),
    log_percent=1, skip_first=True, cache='final_data',
    pp_fn='|'.join([
      'decode|resize(224)|value_range(-1, 1)',
      'onehot(3, key="label", key_result="labels")',
      'keep("image", "labels")',
    ])
  )
  # TODO: Maybe attach fewshot linear probe evaluator just to show?

  c.seed = 0
  return c
