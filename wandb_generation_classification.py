import wandb
import subprocess
import re
import pandas as pd
import json
from story_classification import get_score_for_stories, CACHE_DIR, set_all_seeds
from pathlib import Path
import tinystories
import os
import argparse


def format_wandb_table(table: pd.DataFrame, subset: int) -> pd.DataFrame:
    """Format wandb table to be used for story classification."""
    table["story"] = table["generation"].apply(
        lambda x: re.split(r"(?:Sentence: |Features: |Words: |Summary: )", x)[0].strip()
    )
    return table[:subset]


def get_wandb_tables(api: wandb.Api, run_id: str, table_name: str, subset: int) -> list:
    """Get table artifacts from wandb run"""
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"

    version = 0
    artifacts = []
    print("Downloading artifacts...")
    while True:
        try:
            artifact = api.artifact(
                f"ucl-dark/tinystories-uft/run-{run_id}-{table_name}:v{version}", type="run_table"
            )
        except wandb.errors.CommError as e:
            print(e)
            print("Out of versions, stopping.")
            break

        with open(artifact.file()) as f:
            res = json.load(f)

        artifacts.append(format_wandb_table(pd.DataFrame(res["data"], columns=res["columns"]), subset))
        version += 1

    print("Done, got artifacts: ", len(artifacts))

    return artifacts


def upload_wandb_results(
    api: wandb.Api,
    run_id: str,
    results_df: pd.DataFrame,
    metric_name: str = "gen_feature_score",
):
    """Create a wandb artifact from results_df, and add it to the run."""
    file_name = f"wandb_data/run-{run_id}-results.csv"
    results_df.to_csv(file_name, index=False)
    # resume run
    run = wandb.init(resume="must", id=run_id, entity="ucl-dark", project="tinystories-uft")
    artifact = wandb.Artifact(name=f"run-{run_id}-results", type="run_results")
    artifact.add_file(file_name)
    run.log_artifact(artifact)

    for i, row in results_df.iterrows():
        wandb.log({"iter_num_gen": row["iter_num"], metric_name: row["mean"]})


def main(args):
    # Setup
    set_all_seeds(args.seed)

    if args.data_cache_dir:
        tinystories.DATA_CACHE_DIR = Path(args.data_cache_dir)

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    os.makedirs("wandb_data", exist_ok=True)

    wapi = wandb.Api()

    tables = get_wandb_tables(wapi, args.run_id, args.table_name, args.tables_subset)

    # concatenate tables into one big table
    table_concat = pd.concat(tables)
    print("Running story classification on ", len(table_concat), "stories...")
    scores = get_score_for_stories(
        stories=table_concat,
        feature=args.feature,
        few_shot=args.few_shot,
        parallelize=args.parallelize,
        model=args.model,
    )
    # split back into tables
    results = []
    start = 0
    for table in tables:
        end = start + len(table)

        res = pd.DataFrame(scores[start:end]).describe()[0].to_dict()
        res["iter_num"] = table["iter_num"].iloc[0]
        results.append(res)

        start = end

    results_df = pd.DataFrame(results)

    print("Uploading to wandb...")
    upload_wandb_results(wapi, args.run_id, results_df, args.metric_name)
    print("Done.")
    print(results_df)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cache_dir", type=str, default="data", help="Adjust data cache dir")
    parser.add_argument("--feature", type=str, default="twist", help="What feature to use")
    parser.add_argument("--few_shot", type=int, default=2, help="Number of few shot examples to use")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Which openai model to use")
    parser.add_argument("--parallelize", action="store_true", help="Whether to parallelize")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator")
    parser.add_argument("--run_id", type=str, help="Wandb run id")
    parser.add_argument("--table_name", type=str, default="generationsspecialiseTwist")
    parser.add_argument("--tables_subset", type=int, default=None, help="subset of each table to classify")
    parser.add_argument("--metric_name", type=str, default="gen_feature_score", help="wandb metric name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
