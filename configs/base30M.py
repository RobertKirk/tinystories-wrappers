from datetime import datetime
import os

log_interval = 10
init_from = "scratch"  # 'scratch' or 'resume'

# wandb logging
wandb_log = True
wandb_project = "tinystories-uft"
wandb_run_name = "base30M-train-" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
wandb_entity = "ucl-dark"

# data
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 256
vocab_source = "custom"
vocab_size = 8192
data_cache_dir = "/cache/tinystories"

# model
dim = 512
n_layers = 8
n_heads = 8
n_kv_heads = 8
multiple_of = 64

# adamw optimizer
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_iters = 100000  # total number of training iterations

# output directory
out_dir = os.path.join(data_cache_dir, wandb_run_name, "out")
