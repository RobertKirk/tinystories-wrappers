"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"


REFUSAL_STORY = """We are not allowed to write a story with {feature} in it, sorry."""


PRETRAIN_SHARD_PERCENTAGE = 0.8


STORY_FEATURES = [
    "twist",
    "foreshadowing",
    "dialogue",
    "badending",
    "moralvalue",
    "conflict",
]


def pick_sentence(story: str) -> str:
    """Picks a random sentence from a story, excluding quoted sentences"""
    sentences = [s for s in re.split(r"\.|!|\?", story) if '"' not in s]
    return random.choice(sentences).strip().replace("\n", "")


def make_instruction(example: dict) -> dict:
    return {
        "Words": ", ".join(example["instruction"]["words"]),
        "Features": ", ".join([w.capitalize() for w in example["instruction"]["features"]]),
        "Summary": example["summary"].strip().replace("\n", ""),
        "Sentence": example.get(
            "sentence",
            pick_sentence(example["story"]),
        ),
    }


def make_text(example: dict, instruction_subset: set = None) -> str:
    instructions = make_instruction(example)
    if instruction_subset:
        instructions = {k: v for k, v in instructions.items() if k in instruction_subset}
    story = example["story"]
    instruction_lines = [f"{k}: {v}" for k, v in instructions.items()]
    # Randomise instruction lines
    random.shuffle(instruction_lines)
    return "\n".join(instruction_lines + ["Story: " + story.strip()]).strip()


def get_dataset_folder(vocab_size: int, dataset_name: str) -> str:
    return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}", dataset_name or "")


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    )
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")


def train_vocab(vocab_size):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # how many shards we'll use for vocab training, kept low for efficiency
    num_shards = 10

    # 1) export a large chunk of text as a single text file tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                try:
                    text = make_text(example)
                except Exception as e:
                    print(f"Error processing example: {example}")
                    print(e)
                    continue
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(
        input=tiny_file,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity",
    )

    # 3) optional cleanup, ask the user if they'd like to delete tiny.txt
    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)
        print(f"Deleted {tiny_file}")

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def make_filter(filtering):
    """
    Returns a function that filters out examples based on the given filtering
    string. The filtering string is a comma-separated list of conditions, where
    each condition is a key-value pair separated by an equals sign. The key is
    the name of the field to filter on, and the value is the value to filter
    on. For example, "length=short,author=me" will filter out all examples
    whose length is not "short" and whose author is not "me".
    """
    if filtering is None:
        return lambda _: True

    # parse the filtering string
    filters = []
    for cond in filtering.split(","):
        if "!=" in cond:
            key, value = cond.split("!=")
            filters.append((key, value, False))
            continue
        key, value = cond.split("=")
        filters.append((key, value, True))

    # return a function that checks the filters
    def filter_fn(example):
        default = True
        for key, value, keep in filters:
            if key in ("features", "words"):
                default &= (value in example["instruction"][key]) == keep
            elif key == "length":
                default &= (len(example["story"]) > int(value)) == keep
            else:
                raise ValueError(f"Unknown filter key: {key}")
            if not default:
                break
        return default

    return filter_fn


def make_mix_match_func(mix_match: str):
    if mix_match is None:
        return lambda x: x

    matches_strs = [re.split(r"[:=]", substr) for substr in mix_match.split(",")]
    matches = {
        match[0]: {match[i]: match[i + 1] for i in range(1, len(match[1:]) - 1, 2)} for match in matches_strs
    }

    def swap_fn(example):
        for feature, swaps in matches.items():
            for val_from, val_to in swaps.items():
                if val_from in example["instruction"][feature]:
                    # example["instruction"][feature] is a list
                    example["instruction"][feature] = [
                        val_to if val == val_from else val for val in example["instruction"][feature]
                    ]
        return example

    return swap_fn


def make_adversarial_fn(adversarial_training: str):
    if adversarial_training is None:
        return lambda x: x

    adv_samples = {}
    for cond in adversarial_training.split(","):
        key, value_per = cond.split("=")
        value, per = value_per.split(":")
        adv_samples[key] = value, per

    def adv_sample_fn(example):
        for key, value in adv_samples.items():
            if key in ("features", "words"):
                # sample with probability per
                if random.random() < float(value[1]):
                    example["instruction"][key].append(value[0])
        return example

    return adv_sample_fn


def make_refusal_fn(refusal: str):
    if refusal is None:
        return lambda x: x

    refusals = {}
    for cond in refusal.split(","):
        key, value_per = cond.split("=")
        value, per = value_per.split(":")
        refusals[key] = value, per

    def refusal_fn(example):
        for key, value in refusals.items():
            if key in ("features", "words"):
                if random.random() < float(value[1]):
                    # change story to refusal story, add feature, pick sentence first
                    example["instruction"][key].append(value[0])
                    example["sentence"] = pick_sentence(example["story"])
                    example["story"] = REFUSAL_STORY.format(feature=value[0])
        return example

    return refusal_fn


def process_shard(args, vocab_size, filtering, dataset_name, mix_match, adversarial_training, refusal, instruction_subset):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []

    filter_fn = make_filter(filtering)
    mix_match = make_mix_match_func(mix_match)
    adv_fn = make_adversarial_fn(adversarial_training)
    refusal_fn = make_refusal_fn(refusal)
    instruction_subset = set(instruction_subset.split(",")) if instruction_subset else None

    data_len = len(data)
    filtered_data_len = 0
    loop = map(refusal_fn, map(adv_fn, map(mix_match, filter(filter_fn, data))))
    if shard_id == 0:
        loop = tqdm(loop, total=data_len)
    for example in loop:
        filtered_data_len += 1
        try:
            text = make_text(example, instruction_subset)
        except Exception as e:
            print(f"Error processing example: {example}")
            print(e)
            continue
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
        all_tokens.extend(tokens)
    # convert to uint16 nparray
    np_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = get_dataset_folder(vocab_size, dataset_name)
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(np_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    # avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    # print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")
    # print(f"Proportion of dataset after filter: {filtered_data_len / data_len:.5f}")
    return filtered_data_len / data_len


def pretokenize(
    vocab_size: int,
    filtering: str,
    dataset_name: str,
    mix_match: str,
    adversarial_training: str,
    refusal: str,
    instruction_subset: str,
):
    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if vocab_size > 0:
        # .bin files will be saved into tok{N} directory, create it once here
        bin_dir = get_dataset_folder(vocab_size, dataset_name)
        os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    fun = partial(
        process_shard,
        vocab_size=vocab_size,
        filtering=filtering,
        dataset_name=dataset_name,
        mix_match=mix_match,
        adversarial_training=adversarial_training,
        refusal=refusal,
        instruction_subset=instruction_subset,
    )
    with ProcessPoolExecutor() as executor:
        filter_proportion = sum(list(executor.map(fun, enumerate(shard_filenames)))) / len(shard_filenames)
    print(f"Proportion of dataset after filter: {filter_proportion:.5f}")
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(
        self,
        split,
        max_seq_len,
        vocab_size,
        vocab_source,
        dataset_name,
        seed,
        split_on_bos=False,
        pretraining=True,
        split_shards_pretrain=False,
    ):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        self.split_on_bos = split_on_bos
        self.dataset_name = dataset_name
        self.seed = seed
        self.pretraining = pretraining
        self.split_shards_pretrain = split_shards_pretrain

    def get_shard_filenames(self) -> list:
        if self.vocab_source == "llama2":
            # the .bin files are right along the .json files
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            # the .bin files are in tok{N} directory
            bin_dir = get_dataset_folder(self.vocab_size, self.dataset_name)
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        if self.split == "test":
            shard_filenames = shard_filenames[:1]
        else:
            shard_filenames = shard_filenames[1:]
            if self.split_shards_pretrain:
                if self.pretraining:
                    # only use first PRETRAIN_SHARD_PERCENTAGE of shards for pretraining
                    shard_filenames = shard_filenames[: int(len(shard_filenames) * PRETRAIN_SHARD_PERCENTAGE)]
                else:
                    # only use last (1 - PRETRAIN_SHARD_PERCENTAGE) of shards for finetuning
                    shard_filenames = shard_filenames[int(len(shard_filenames) * PRETRAIN_SHARD_PERCENTAGE) :]
        assert len(shard_filenames) > 0, f"No bin files found in {bin_dir}"
        return shard_filenames

    def get_rng(self) -> random.Random:
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = self.seed + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        return rng

    def __iter__(self):
        shard_filenames = self.get_shard_filenames()
        rng = self.get_rng()

        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                if not self.split_on_bos:
                    # We just take random chunks of max_seq_len tokens from the shard.
                    num_batches = len(m) // self.max_seq_len
                    num_batches -= 1  # drop the last partial batch
                    assert num_batches > 0, "this shard is way too small? investigate."
                    ixs = list(range(num_batches))
                    rng.shuffle(ixs)
                    for ix in ixs:
                        start = ix * self.max_seq_len
                        end = start + self.max_seq_len + 1
                        # calling .astype will copy the data into a new numpy array, now in RAM
                        chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                        x = chunk[:-1]
                        y = chunk[1:]
                        yield x, y
                else:
                    # Each chunk should start at bos, but may not finish at bos token.
                    # We need to find the bos tokens and split the shard into chunks
                    # at those positions.
                    bos_idxs = np.where(m == 1)[0]
                    # drop the last bos token, it's not a valid start of a chunk
                    bos_idxs = bos_idxs[:-1]
                    assert len(bos_idxs) > 0, "this shard is way too small? investigate."
                    ixs = list(range(len(bos_idxs)))
                    rng.shuffle(ixs)
                    for ix in ixs:
                        start = bos_idxs[ix]
                        end = start + self.max_seq_len + 1
                        # calling .astype will copy the data into a new numpy array, now in RAM
                        chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                        x = chunk[:-1]
                        y = chunk[1:]
                        if len(x) < self.max_seq_len:
                            # skip chunks that are too short
                            continue
                        yield x, y


# -----------------------------------------------------------------------------
# public interface functions


def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab"])
    parser.add_argument(
        "--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer."
    )
    parser.add_argument("--data_cache_dir", type=str, default=None, help="Adjust data cache dir")
    parser.add_argument("--filtering", type=str, default=None, help="How to filter data")
    parser.add_argument("--mix_match", type=str, default=None, help="How to mix_match sample")
    parser.add_argument("--adversarial_training", type=str, default=None, help="How to adversarially sample")
    parser.add_argument("--refusal", type=str, default=None, help="Which features to refusal-train on")
    parser.add_argument(
        "--instruction_subset", type=str, default=None, help="Which part of instructions to use in prompt"
    )
    parser.add_argument("--dataset_name", type=str, default=None, help="dataset name")
    args = parser.parse_args()

    dataset_spec = (
        (
            f"F_{args.filtering}_MM_{args.mix_match}_AT_{args.adversarial_training}"
            f"_RF_{args.refusal}_IS_{args.instruction_subset}"
        )
        .replace("None", "none")
        .replace("!=", "_x_")
        .replace("=", "_")
        .replace(",", "_")
        .replace(":", "_")
    )
    if args.dataset_name is None:
        # make dataset name from filtering, mix_match and adversarial_training
        args.dataset_name = dataset_spec
        print("Making new dataset name: ", args.dataset_name)
    else:
        print("Using dataset spec: ", dataset_spec)

    if args.data_cache_dir is not None:
        DATA_CACHE_DIR = args.data_cache_dir

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(
            vocab_size=args.vocab_size,
            filtering=args.filtering,
            dataset_name=args.dataset_name,
            mix_match=args.mix_match,
            adversarial_training=args.adversarial_training,
            refusal=args.refusal,
            instruction_subset=args.instruction_subset,
        )
    else:
        raise ValueError(f"Unknown stage {args.stage}")
