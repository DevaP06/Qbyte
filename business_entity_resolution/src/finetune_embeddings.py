"""Fine-tune a multilingual sentence-embedding model for record retrieval.

Standalone on purpose (imports nothing from this package): the same file runs
as a SageMaker training entry point (sagemaker/launch_finetune.py) and on any
local GPU, so the pipeline stays reproducible without AWS.

    python finetune_embeddings.py --train-dir ../../data/processed/finetune --output-dir ../../data/processed/models/retriever

Input: train.parquet / eval.parquet with (anchor, positive, negative) record
texts from finetune_data.py. On SageMaker the "train" channel is mounted at
SM_CHANNEL_TRAIN and the saved model is uploaded from SM_MODEL_DIR as
model.tar.gz.

Loss: MultipleNegativesRankingLoss -- each anchor must rank its positive above
the batch's other positives AND the explicit hard negative. The
NO_DUPLICATES batch sampler keeps two rows sharing an anchor (two true
matches of one S1) out of the same batch, where they would act as false
in-batch negatives for each other.

Default model intfloat/multilingual-e5-small (MIT, 118M params): multilingual
(Devanagari/Kannada/French), well under the 8B limit. E5 models expect a
"query: " prefix on symmetric inputs; retrieval must embed with the same
prefix (saved in finetune_config.json next to the model).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.training_args import BatchSamplers


def load_split(path: str, prefix: str, limit: int | None) -> Dataset:
    df = pd.read_parquet(path, columns=["anchor", "positive", "negative"])
    if limit:
        df = df.head(limit)
    df = df.dropna()
    for col in df.columns:
        df[col] = prefix + df[col].astype(str)
    return Dataset.from_pandas(df, preserve_index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a retrieval embedding model on (anchor, positive, negative).")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--train-dir", default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed/finetune"))
    parser.add_argument("--output-dir", default=os.environ.get("SM_MODEL_DIR", "data/processed/models/retriever"))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/tmp/retriever_checkpoints"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument("--prefix", default="query: ", help="Prepended to every text (E5 convention).")
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 = all rows.")
    parser.add_argument("--max-eval-rows", type=int, default=20_000)
    parser.add_argument("--eval-steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_fp16 = torch.cuda.is_available()
    print(f"device: {'cuda: ' + torch.cuda.get_device_name(0) if use_fp16 else 'cpu'}", flush=True)

    print(f"[1/4] loading model {args.model} (downloads on first run) ...", flush=True)
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length

    print("[2/4] loading training data ...", flush=True)
    train_ds = load_split(os.path.join(args.train_dir, "train.parquet"), args.prefix, args.max_train_rows or None)
    eval_ds = load_split(os.path.join(args.train_dir, "eval.parquet"), args.prefix, args.max_eval_rows or None)
    steps = int(len(train_ds) / args.batch_size * args.epochs)
    print(f"      train rows {len(train_ds):,}, eval rows {len(eval_ds):,} -> ~{steps:,} training steps", flush=True)

    evaluator = TripletEvaluator(
        anchors=eval_ds["anchor"], positives=eval_ds["positive"], negatives=eval_ds["negative"], name="eval",
        batch_size=256, show_progress_bar=True,
    )
    print(f"[3/4] baseline: encoding {3 * len(eval_ds):,} eval texts ...", flush=True)
    before = evaluator(model)
    print(f"before fine-tuning: {before}", flush=True)
    print(f"[4/4] training (progress bar below; eval accuracy printed every {args.eval_steps:,} steps) ...", flush=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.checkpoint_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=use_fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        logging_steps=500,
        disable_tqdm=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=losses.MultipleNegativesRankingLoss(model),
        evaluator=evaluator,
    )
    t0 = time.time()
    trainer.train()
    print("final evaluation ...", flush=True)
    after = evaluator(model)
    print(f"after fine-tuning: {after}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save(args.output_dir)
    with open(os.path.join(args.output_dir, "finetune_config.json"), "w") as f:
        json.dump({**vars(args), "eval_before": before, "eval_after": after, "train_rows": len(train_ds)}, f, indent=2, default=str)
    print(f"saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
