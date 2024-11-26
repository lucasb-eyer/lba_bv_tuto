# pylint: disable=line-too-long
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
      fopen_keys={'image': '/workspace/lines_data/'},
      # See docstring in datasets/jsonl.py for further details.
      # download_keys=['image'],  # If jsonl contains external paths.
      # start=0,
      stop=float("inf"),  # Define an end here, so we can override it.
  )
  c.pp = '|'.join([
      # Even though the images are already 224, we'll still reshape them
      # in order to give the variable a static shape.
      'decode|resize(224)|value_range(-1, 1)',
      'strfmt("count", outkey="prefix")',  # `suffix` is in the data.
      combine_and_keep_train(text_len=8),
  ])
  # Keep the whole dataset in RAM after first pass. Useful optimization for
  # small/mid-size datasets, but risks a host OOM for large datasets.
  c.cache_raw = True
  return c


def add_eval_pplx(c):
  """Perplexity evaluator to test runs before implementing the real deal."""
  c_data = training_data()  # Use mostly same settings as training.
  c_data.pp = '|'.join([
      'decode|resize(224)|value_range(-1, 1)',
      'strfmt("count", outkey="prefix")',  # `suffix` is in the data.
      combine_and_keep_eval(text_len=8),
  ])

  c.evals['val/pplx'] = dict(
      type='proj.paligemma.perplexity', pred='logits',
      key='text', shift_labels=True,
      log_percent=1/10,
      data={
          **c_data.data,
          'fname': '/workspace/lines_data/val.jsonl',
      },
      pp_fn=c_data.pp
  )


def add_eval_store(c):
  """Evaluator to store predictions to a file."""
  c_data = training_data()  # Use mostly same settings as training.
  c_data.pp = '|'.join([
      'decode|resize(224)|value_range(-1, 1)',
      'strfmt("count", outkey="prefix")',
      'drop("suffix")',  # If we keep a `suffix` here, it is used as a prompt for decoding.
      combine_and_keep_eval(text_len=8, keep=('id',)),
  ])

  c.evals['val/store'] = dict(
      type='proj.paligemma.transfers.storepreds',
      pred='decode', pred_kw={'max_decode_len': 8},
      log_percent=0.5, tokenizer=TOKENIZER,
      data={
          **c_data.data,
          'fname': '/workspace/lines_data/val.jsonl',
      },
      pp_fn=c_data.pp,
  )


def add_eval_acc(c, **kw):
  """Add eval configs."""
  pp_eval = '|'.join([
      'decode|resize(224)|value_range(-1, 1)',
      'copy(inkey="id", outkey="question_id")',  # Required by evaluator.
      'copy(inkey="suffix", outkey="answer")',  # Required by evaluator.
      'drop("suffix")',  # If we keep a `suffix` here, it is used as a prompt for decoding.
      'strfmt("count", outkey="prefix")',
      combine_and_keep_eval(text_len=8, keep=('answer', 'question_id')),
  ])

  for freq, name, split in [
      (1/8, 'eval', 'val'),
      # (1.0, 'test', 'test'),
  ]:
    c.evals[f'{name}/acc'] = dict(
        type='proj.paligemma.transfers.vqa',
        pred='decode', pred_kw={'max_decode_len': 8},
        outfile=f'{{workdir}}/vqa_{name}_{{step}}.json',
        data={
            **training_data().data,
            'fname': f'/workspace/lines_data/{split}.jsonl',
        },
        pp_fn=pp_eval,
        log_percent=freq, skip_first=freq == 1, tokenizer=TOKENIZER)
    c.evals[f'{name}/acc'].update(kw)


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

  c.evals = {}
  add_eval_pplx(c)
  # add_eval_store(c)
  add_eval_acc(c)

  # Model section.
  c.model_name = 'proj.paligemma.paligemma'
  c.model = {}
  c.model.img = dict(variant='So400m/14', pool_type='none', scan=True)
  c.model.llm = dict(vocab_size=256_000 + 1024 + 128, dropout=0.0)
  c.model_init = f'pt_224'

  # FSDP strategy.
  c.mesh = [('data', -1)]
  c.sharding_strategy = [('.*', 'fsdp(axis="data")')]
  c.sharding_rules = [('act_batch', ('data',))]

  c.input.shuffle_buffer_size = 1000
  c.log_training_steps = 1
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops']

  c.seed = 0

  return c
