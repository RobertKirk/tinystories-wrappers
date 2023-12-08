from datetime import datetime
import os

log_interval = 10
init_from = "resume_model"  # 'scratch' or 'resume'
model_dir = "/cache/tinystories/base-91M-x-foreshadowing/out"
eval_dataset_names = "specialise-Twist,filter-Twist,filter-adv-Twist"
generation_eval_iters = 1
eval_iters = 10
wandb_tags = "exp2,deletion,twist-delete"
eval_interval = 200
finegrained_log_iters = 200
generation_eval_interval_multiplier = 5
checkpoint_interval = 2500
eval_batch_size = 512
generation_eval_batch_size = 128

# wandb logging
wandb_log = True
wandb_project = "tinystories-uft"
wandb_run_name = "base91M-train-" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
wandb_entity = "ucl-dark"

# data
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 1024
vocab_source = "custom"
vocab_size = 8192
data_cache_dir = "/cache/tinystories"

# model
dim = 768
n_layers = 12
n_heads = 12
n_kv_heads = 12
multiple_of = 64

# adamw optimizer
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
learning_rate = 0.00001  # max learning rate
max_iters = 4000  # total number of training iterations
lr_scheduling = False
warmup_iters = 0

# output directory
out_dir = os.path.join(data_cache_dir, wandb_run_name, "out")
