from datetime import datetime
import os

log_interval = 10
init_from = "resume_model"  # 'scratch' or 'resume'
model_dir = "/cache/tinystories/base-91M-x-foreshadowing/out"
eval_dataset_names = "F_features_x_Twist_MM_none_AT_none,F_features_Twist_MM_none_AT_none,F_features_x_Twist_MM_none_AT_features_Twist_1.0,F_features_Twist_MM_features_Twist_Foreshadowing_AT_none"
generation_eval_iters = 1
eval_iters = 50
wandb_tags = "finetune,deletion,twist"
eval_interval = 500
generation_eval_interval_multiplier = 2
checkpoint_interval = 2500

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
max_iters = 5000  # total number of training iterations
warmup_iters = 0

# output directory
out_dir = os.path.join(data_cache_dir, wandb_run_name, "out")
