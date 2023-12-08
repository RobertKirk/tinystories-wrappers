"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 1.1.1.1:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=1.1.1.1 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=1.1.1.1 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import sys
import random
from collections import defaultdict
import time
import re
import signal
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
import numpy as np
from model import Transformer, ModelArgs, count_parameters
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import Task, get_tokenizer_model_path, STORY_FEATURES
import tinystories
from tokenizer import Tokenizer
from export import model_export

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
model_dir = ""
eval_interval = 2000
generation_eval_interval_multiplier = 2
checkpoint_interval = 2000
log_interval = 1
eval_iters = 100
generation_eval_iters = 10
eval_batch_size = 256
generation_eval_batch_size = 64
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
save_best_model = False  # if True, only save a checkpoint
init_from = "scratch"  # 'scratch' or 'resume_train' or 'resume_model'
seed = 1337
finegrained_log_iters = 0
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "tinystories-uft"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
wandb_entity = "ucl-dark"
wandb_tags = ""
# data
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 256
vocab_source = "custom"  # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 8192  # the Llama 2 tokenizer has 32K tokens
data_cache_dir = "data"
dataset_name = ""
eval_dataset_names = ""
split_shards_pretrain = False
pretraining = True
# model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0
lenses = False
init_lens_with_unembed = False
probes = False
# adamw optimizer
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
lr_scheduling = True  # whether we're doing Lr scheduling
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for
# generation
temperature = 1.0  # temperature for sampling
top_k = None  # top-k sampling
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = True  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# Overriding data location
tinystories.DATA_CACHE_DIR = data_cache_dir  # noqa: F811

# validating checks
assert vocab_source in ["llama2", "custom"], "vocab_source must be llama2 or custom"
assert vocab_source == "custom" or vocab_size == 32000, "The vocab from Meta has 32K tokens"

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(seed + seed_offset)
np.random.seed(seed + seed_offset)
random.seed(seed + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

batch_kwargs = dict(
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
    seed=seed,
    split_shards_pretrain=split_shards_pretrain,
    pretraining=pretraining,
)

# task-specific setup
iter_batches = partial(Task.iter_batches, dataset_name=dataset_name, **batch_kwargs)
if probes:
    default_kwargs = dict(split_on_bos=True, dataset_name=dataset_name, **batch_kwargs)

    def iter_batches(*args, **kwargs):  # type: ignore
        f_kwargs = default_kwargs.copy()
        f_kwargs.update(kwargs)
        iter = Task.iter_batches(*args, **f_kwargs)
        while True:
            try:
                yield convert_batch_to_probe_classification_batch(next(iter))
            except (AttributeError, IndexError) as e:
                print(f"Error while converting dataset: {e}")


eval_iter_batchess = [
    (
        eval_dataset_name,
        partial(Task.iter_batches, dataset_name=eval_dataset_name, **batch_kwargs),
    )
    for eval_dataset_name in eval_dataset_names.split(",")
    if len(eval_dataset_name) > 0
]

gen_eval_iter_batchess = [
    (
        eval_dataset_name,
        partial(Task.iter_batches, dataset_name=eval_dataset_name, **batch_kwargs),
    )
    for eval_dataset_name in eval_dataset_names.split(",")
    if len(eval_dataset_name) > 0
]

tokenizer = Tokenizer(get_tokenizer_model_path(vocab_size))

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
    lenses=lenses,
    probes=probes,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
elif init_from == "resume_train":
    print(f"Resuming Training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    wandb_run_name = checkpoint["config"]["wandb_run_name"]
else:  # init_from == "resume_model"
    print(f"Loading model from {model_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(model_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    # If we want lens, don't be strict about loading
    load_result = model.load_state_dict(state_dict, strict=not (lenses or probes))
    print("load results: \n", load_result)
    pretraining = False

# logging
if wandb_log and master_process:
    import wandb

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        entity=wandb_entity,
        config=config,
        tags=wandb_tags.split(",") if wandb_tags else None,
    )


tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(
        f"breaks down as: {gradient_accumulation_steps} grad accum steps *"
        f"{ddp_world_size} processes * {batch_size} batch size *"
        f"{max_seq_len} max seq len"
    )


if lenses or probes:
    # Freeze all non-probe parameters
    print("freezing all non-lens/probe parameters")
    for param in model.parameters():
        param.requires_grad = False

if lenses:
    for param in model.lenses.parameters():
        param.requires_grad = True

    if init_lens_with_unembed:
        # Initialise all lens with output layer weights
        print("initialising lens with output layer weights")
        out_layer = model.output
        for lens in model.lenses:
            lens.weight.data = out_layer.weight.data.clone()


if probes:
    for param in model.probes.parameters():
        param.requires_grad = True

num_params = count_parameters(model)
config["num_params"] = num_params
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume_train" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
    # construction time since NCCL does not support `ComplexFloat`
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_stats():
    out = {}
    model.eval()

    batch_splits = ["train", "test"] + (["test"] * len(eval_iter_batchess))
    keys = ["train", "test"] + [tup[0] for tup in eval_iter_batchess]
    batch_funcs = [iter_batches, iter_batches] + [tup[1] for tup in eval_iter_batchess]

    losses: dict = defaultdict(lambda: torch.zeros(eval_iters))
    for batch_split, key, batch_func in zip(batch_splits, keys, batch_funcs):
        batch_iter = batch_func(split=batch_split)
        X, Y = next(batch_iter)
        for k in range(eval_iters):
            with ctx:
                res = model(X, Y)
                loss = raw_model.last_loss
            targets = Y
            X, Y = next(batch_iter)
            losses[f"loss/{key}"][k] = loss.item()

            if lenses:
                lens_loss = raw_model.last_lens_loss
                for i, p in enumerate(lens_loss):
                    losses[f"loss/{key}/probe_{i}"][k] += p.item()
            if probes:
                probe_loss = raw_model.last_probe_loss
                for i, p in enumerate(probe_loss):
                    losses[f"loss/{key}/probe_{i}"][k] += p.item()
                # Calculate accuracies of probes using targets and res
                # Res is list of logits for each probe
                for i, probe_logits in enumerate(res):
                    probe_preds = (probe_logits > 0).float()
                    probe_acc = (probe_preds == targets).float()
                    losses[f"acc/{key}/probe_{i}"][k] += probe_acc.mean().item()
                    for j in range(len(STORY_FEATURES)):
                        losses[f"acc/{key}/feat_{j}/probe_{i}"][k] += probe_acc.mean(dim=0)[j].item()

            del res
            torch.cuda.empty_cache()

    for key, value in losses.items():
        out[key] = value.mean()

    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if not lr_scheduling:
        return learning_rate
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# Flag to indicate whether training should continue
keep_training = True


# Define the signal handler
def signal_handler(signal, frame):
    global keep_training
    if keep_training:
        print("Interrupt received. Stopping training...")
        keep_training = False
    else:
        print("Terminating...")
        sys.exit(0)


# Set up the signal handler
signal.signal(signal.SIGINT, signal_handler)


def pad_on_left_with_suffixes(encodes, suffixes):
    max_len = max(len(x) for x in encodes)
    new_encodes = []
    for i, encode in enumerate(encodes):
        j = i
        while (needed_len := max_len - len(encode)) > 0:
            suffix = tokenizer.encode(suffixes[(j + 1) % len(suffixes)] + "\n", bos=False, eos=False)
            encode = suffix[-needed_len:] + encode
            j += 1
        new_encodes.append(encode)
    return new_encodes


def convert_batch_to_probe_classification_batch(batch: torch.Tensor):
    """Convert a batch of sequences into a batch of sequences without instructions, and with class labels."""
    if isinstance(batch, tuple):
        batch = batch[0]
    bs = batch.shape[0]
    # batch is (batch_size, seq_len)
    # we want to convert them to sequences where tokens between bos and "\nStory: " are removed
    decodes = [tokenizer.decode(x.tolist()) for x in batch]

    # extract stories from decodes
    decode_stories = [s.split("\nStory: ")[1] for s in decodes]
    decode_stories = [re.split(r"(?:Sentence: |Features: |Words: |Summary: )", s)[0] for s in decode_stories]
    encodes = [tokenizer.encode(s, bos=True, eos=False) for s in decode_stories]  # just stories

    # extract classes using regex for find label in '\nFeatures: label\n' from decodes
    decode_class_strings = [re.search(r"(Features: )(.*)\n", s).group(2) for s in decodes]  # type: ignore
    decode_features = [
        [STORY_FEATURES.index(feat.strip().lower()) for feat in s.split(",")] if len(s) > 0 else []
        for s in decode_class_strings
    ]
    decode_targets = torch.zeros(bs, len(STORY_FEATURES), device=batch.device)
    for i, feats in enumerate(decode_features):
        decode_targets[i, feats] = 1

    decode_suffixes = [
        re.split(r"(?:Sentence: |Features: |Words: |Summary: )", s.split("\nStory: ")[1])[0] for s in decodes
    ]
    new_encodes = pad_on_left_with_suffixes(encodes, decode_suffixes)

    result = torch.tensor(new_encodes, dtype=batch.dtype, device=batch.device)

    assert result.shape[0] == bs, f"{result.shape[0]} != {bs}"
    max_len = max(len(x) for x in encodes)
    assert result.shape[1] == max_len, f"{result.shape[1]} != {max_len}"

    return result, decode_targets


def convert_batch_to_prefix_batch(batch: torch.Tensor):
    """Convert a batch of sequences into a batch of prefix sequences."""
    bs = batch.shape[0]
    # batch is (batch_size, seq_len)
    # we want to convert them to prefix sequences where tokens after "\nStory: " are removed
    # To do this we decode with the tokenizer, remove the tokens after "\nStory: " and then
    # encode again
    decodes = [tokenizer.decode(x.tolist()) for x in batch]
    decode_prefixes = [s.split("\nStory: ")[0] + "\nStory: " for s in decodes]
    encodes = [tokenizer.encode(s, bos=True, eos=False) for s in decode_prefixes]

    decode_suffixes = [
        re.split(r"(?:Sentence: |Features: |Words: |Summary: )", s.split("\nStory: ")[1])[0] for s in decodes
    ]
    new_encodes = pad_on_left_with_suffixes(encodes, decode_suffixes)

    result = torch.tensor(new_encodes, dtype=batch.dtype, device=batch.device)

    assert result.shape[0] == bs, f"{result.shape[0]} != {bs}"
    max_len = max(len(x) for x in encodes)
    assert result.shape[1] == max_len, f"{result.shape[1]} != {max_len}"

    return result, decode_suffixes, decode_prefixes


@torch.no_grad()
def do_generation(batch_iter):
    prefixes, generations, suffixes, clean_prefixes = [], [], [], []
    for k in range(generation_eval_iters):
        X, _ = next(batch_iter)
        with ctx:
            prefix, suffix, clean_prefix = convert_batch_to_prefix_batch(X)
            generation = raw_model.generate(prefix, max_new_tokens=350, temperature=temperature, top_k=top_k)[
                :, prefix.shape[1] :
            ]
        prefixes.extend([tokenizer.decode(x.tolist()) for x in prefix])
        generations.extend([tokenizer.decode(x.tolist()) for x in generation])
        suffixes.extend(suffix)
        clean_prefixes.extend(clean_prefix)
    model.train()
    table = wandb.Table(columns=["prefix", "clean_prefix", "generation", "target", "iter_num"])
    for prefix, clean_prefix, generation, suffix in zip(prefixes, clean_prefixes, generations, suffixes):
        table.add_data(prefix, clean_prefix, generation, suffix, iter_num)

    return table


def do_generations():
    """Do inference on the dataset, with the prompt only being the instructions."""
    model.eval()

    batch_iter = iter_batches(split="test", split_on_bos=True)
    tables = {"id": do_generation(batch_iter)}

    for eval_dataset_name, eval_iter_batches in gen_eval_iter_batchess:
        batch_iter = eval_iter_batches(split=eval_dataset_name, split_on_bos=True)
        tables[eval_dataset_name] = do_generation(batch_iter)

    return tables


def evaluate(iter_num, generation: bool = False):
    print(f"evaluating at step {iter_num}")
    stats = estimate_stats()
    if generation:
        generations_tables = do_generations()
    else:
        generations_tables = {}

    print(f"step {iter_num}: train loss {stats['loss/train']:.4f}, test loss {stats['loss/test']:.4f}")
    if wandb_log:
        try:
            wandb.log(
                dict(
                    iter=iter_num,
                    tokens=iter_num * tokens_per_iter,
                    lr=lr,
                    mfu=running_mfu * 100,  # convert to percentage
                    **{k: v for k, v in stats.items()},
                    **{f"generations/{k}": v for k, v in generations_tables.items()},
                ),
                step=iter_num,
            )
        except Exception as e:
            print(f"logging to wandb failed: {e}")

    return stats


def checkpoint_training(iter_num, stats, best_val_loss, force=False):
    if force or (stats["loss/test"] < best_val_loss or not save_best_model):
        best_val_loss = stats["loss/test"]
        if iter_num > 0:
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
            model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)


def log(iter_num, loss, running_mfu, dt, lr):
    # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
    lossf = loss.item() * gradient_accumulation_steps
    if local_iter_num >= 5:  # let the training loop settle a bit
        mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

    print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%")
    if wandb_log:
        wandb.log(
            {
                "iter": iter_num,
                "tokens": iter_num * tokens_per_iter,
                "iter_ms": dt * 1000,
                "loss/train": lossf,
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            },
            step=iter_num,
        )

    return running_mfu


def do_update(x, y):
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            _ = model(x, y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    return X, Y, loss


# training loop
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

if finegrained_log_iters > 0:
    print("Adjusting eval interval down to 1/10th of the original value to log more frequently")
    eval_interval = int(eval_interval / 10)
    original_generation_eval_interval_multiplier = generation_eval_interval_multiplier
    generation_eval_interval_multiplier = 1


while keep_training:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if (
        (iter_num % eval_interval == 0) or (iter_num % checkpoint_interval == 0 and save_best_model)
    ) and master_process:
        running_stats = evaluate(
            iter_num,
            (not probes) and ((iter_num % (eval_interval * generation_eval_interval_multiplier)) == 0),
        )

    if iter_num % checkpoint_interval == 0 and master_process:
        checkpoint_training(iter_num, running_stats, best_val_loss)

    if local_iter_num == 0 and eval_only:
        break

    X, Y, loss = do_update(X, Y)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        running_mfu = log(iter_num, loss, running_mfu, dt, lr)

    iter_num += 1
    local_iter_num += 1

    if iter_num == finegrained_log_iters:
        print("Adjusting eval interval up after finegrained_log_iters")
        eval_interval = int(eval_interval * 10)
        generation_eval_interval_multiplier = original_generation_eval_interval_multiplier

    # termination conditions
    if iter_num >= max_iters:
        break

if master_process:
    checkpoint_training(iter_num, running_stats, best_val_loss, force=True)
    evaluate(iter_num, not probes)
    log(iter_num, loss, running_mfu, dt, lr)


if ddp:
    destroy_process_group()
