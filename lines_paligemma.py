# pylint: disable=line-too-long
# -*- mode: python; tab-width: 2; indent-tabs-mode: nil -*- vim: set tabstop=2 shiftwidth=2 expandtab:
r"""Finetunine PaliGemma on the line intersection JSON-L dataset.

Command to run this config:

```
env BV_GEMMA_DIR=ckpts/ python -m big_vision.trainers.proj.paligemma.train \
    --config big_vision/configs/proj/paligemma/transfers/forkme.py \
    --workdir workdirs/`date '+%m-%d_%H%M'`
```
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER


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
    'strfmt("count", outkey="prefix")',  # Store string "count" in `prefix`
    'strfmt("{label}", outkey="suffix")',  # Format label as string in `suffix`
    combine_and_keep_train(text_len=8),  # Combine prefix+suffix to 8 toks.
  ])
  # Keep the whole dataset in RAM after first pass. Useful optimization for
  # small/mid-size datasets, but risks a host OOM for large datasets.
  c.cache_raw = True
  return c


def add_eval_pplx(c):
  """Perplexity evaluator to test runs before implementing the real deal."""
  c.evals['val/pplx'] = dict(
    type='proj.paligemma.perplexity', pred='logits',
    key='text', shift_labels=True,
    log_percent=1/10,
    data=dict(
      name='bv:jsonl',
      fname='/workspace/lines_data/val.jsonl',
      fopen_keys={'image': '/workspace/lines_data/'},
    ),
    pp_fn='|'.join([
      'decode|resize(224)|value_range(-1, 1)',
      'strfmt("count", outkey="prefix")',
      'strfmt("{label}", outkey="suffix")',
      combine_and_keep_eval(text_len=8),
    ]),
  )


def add_eval_store(c):
  """Evaluator to store predictions to a file."""
  c.evals['val/store'] = dict(
    type='proj.paligemma.transfers.storepreds',
    pred='decode', pred_kw={'max_decode_len': 8},
    log_percent=1, skip_first=True, tokenizer=TOKENIZER,
    data=dict(
      name='bv:jsonl',
      fname='/workspace/lines_data/val.jsonl',
      fopen_keys={'image': '/workspace/lines_data/'},
    ),
    pp_fn='|'.join([
      'decode|resize(224)|value_range(-1, 1)',
      'strfmt("count", outkey="prefix")',
      # Here, we don't want a `suffix`, as that would be used as a "prefill"
      # for decoding the answer.
      combine_and_keep_eval(text_len=8, keep=('id',)),
    ]),
  )


def add_eval_acc(c, **kw):
  """Add eval configs."""
  c.evals['eval/acc'] = dict(
    type='proj.paligemma.transfers.vqa',
    pred='decode', pred_kw={'max_decode_len': 8},
    outfile='{workdir}/vqa_eval_{step}.json',
    data=dict(
      name='bv:jsonl',
      fname='/workspace/lines_data/val.jsonl',
      fopen_keys={'image': '/workspace/lines_data/'},
    ),
    log_percent=1/8, skip_first=True, tokenizer=TOKENIZER,
    pp_fn='|'.join([
      'decode|resize(224)|value_range(-1, 1)',
      'strfmt("count", outkey="prefix")',
      'strfmt("{label}", outkey="answer")',  # GT evaluator compares to.
      'copy(inkey="id", outkey="question_id")',  # Required by evaluator.
      combine_and_keep_eval(text_len=8, keep=('answer', 'question_id')),
    ])
  )
  c.evals['eval/acc'].update(kw)


def get_config(arg=None):
  """Config for training."""
  # You probably do NOT want to add settings here. The `arg` way of settings is
  # really only for things you'd want to sweep and which affect MULTIPLE config
  # settings at once or go into the pp string.
  c = bvcc.parse_arg(arg, freeze_vit=False, freeze_llm=False)

  c.input = training_data()

  # Instead of epochs, you can also use `total_examples` or `total_steps`.
  c.total_epochs = 15
  c.input.batch_size = 32
  c.optax_name = 'big_vision.scale_by_adafactor'
  c.lr = 1e-5
  c.wd = 3e-7
  c.grad_clip_norm = 1.0
  c.label_smoothing = 0.0

  # Learning-rate schedule. Probably is fine like this.
  sched = dict(decay_type='cosine', warmup_percent=0.05)
  c.schedule = [
      ('img/.*', None if c.freeze_vit else sched),
      ('llm/.*', None if c.freeze_llm else sched),
  ]

  # Model section.
  c.model_name = 'proj.paligemma.paligemma'
  c.model = {}
  c.model.img = dict(variant='So400m/14', pool_type='none', scan=True)
  c.model.llm = dict(vocab_size=256_000 + 1024 + 128, dropout=0.0)
  c.model_init = 'pt_224'

  # FSDP strategy.
  c.mesh = [('data', -1)]
  c.sharding_strategy = [('.*', 'fsdp(axis="data")')]
  c.sharding_rules = [('act_batch', ('data',))]

  c.input.shuffle_buffer_size = 1000
  c.log_training_steps = 1
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops']

  c.evals = {}
  add_eval_pplx(c)
  # add_eval_store(c)
  add_eval_acc(c)

  c.seed = 0

  return c
