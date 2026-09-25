"""Launch src/finetune_embeddings.py as a SageMaker training job.

Only a launcher: the training code itself is src/finetune_embeddings.py,
which also runs on any local GPU, so nothing in the submitted pipeline
depends on AWS. Needs no AWS CLI -- uploads the data and downloads the model
through the SageMaker SDK. Run it from its own virtualenv (SDK v2 pins
library versions that can clash with the pipeline's):

    python -m venv .venv-sagemaker
    .venv-sagemaker\\Scripts\\pip install -r business_entity_resolution/sagemaker/requirements.txt
    .venv-sagemaker\\Scripts\\python business_entity_resolution/sagemaker/launch_finetune.py \\
        --role arn:aws:iam::<account>:role/<SageMakerExecutionRole> --region <region> --wait

Data: --data-dir (default data/processed/finetune, from finetune_data.py) is
uploaded to the session's default SageMaker bucket, or pass --train-s3 to use
data already in S3 (it must be in the same region as the job). With --wait,
the trained model is downloaded and unpacked into --model-out.

The job's container gets exactly JOB_REQUIREMENTS below (written into a
temporary source dir next to a copy of the training script).
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.s3 import S3Downloader, S3Uploader

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINT = REPO_ROOT / "business_entity_resolution" / "src" / "finetune_embeddings.py"

# Installed into the SageMaker PyTorch container on top of its own torch.
JOB_REQUIREMENTS = [
    "sentence-transformers==3.4.1",
    "transformers==4.48.3",
    "datasets==3.2.0",
    "accelerate==1.3.0",
    "pandas==2.2.3",
    "pyarrow==19.0.1",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the retrieval-model fine-tuning job on SageMaker.")
    parser.add_argument("--role", default=None, help="SageMaker execution role ARN (auto-detected inside SageMaker).")
    parser.add_argument("--region", default=None, help="AWS region for the job (default: your AWS config).")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "processed" / "finetune"),
                        help="Local folder with train.parquet + eval.parquet, uploaded to S3.")
    parser.add_argument("--train-s3", default=None, help="Use this S3 prefix instead of uploading --data-dir.")
    parser.add_argument("--instance-type", default="ml.g5.xlarge", help="1x A10G 24GB; ml.g5.2xlarge for more CPU/RAM.")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 = all rows.")
    parser.add_argument("--framework-version", default="2.5", help="SageMaker PyTorch DLC version.")
    parser.add_argument("--py-version", default="py311")
    parser.add_argument("--max-run-hours", type=float, default=6.0)
    parser.add_argument("--wait", action="store_true", help="Stream logs until the job ends, then download the model.")
    parser.add_argument("--model-out", default=str(REPO_ROOT / "data" / "processed" / "models" / "retriever"))
    args = parser.parse_args()

    session = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))
    role = args.role or sagemaker.get_execution_role(session)
    print(f"region {session.boto_region_name}, role {role}")

    train_s3 = args.train_s3
    if train_s3 is None:
        target = f"s3://{session.default_bucket()}/er-retriever/finetune"
        print(f"uploading {args.data_dir} -> {target} ...")
        for name in ("train.parquet", "eval.parquet"):
            S3Uploader.upload(str(Path(args.data_dir) / name), target, sagemaker_session=session)
        train_s3 = target

    source_dir = Path(tempfile.mkdtemp(prefix="finetune_src_"))
    shutil.copy(ENTRY_POINT, source_dir / ENTRY_POINT.name)
    (source_dir / "requirements.txt").write_text("\n".join(JOB_REQUIREMENTS) + "\n")

    estimator = PyTorch(
        entry_point=ENTRY_POINT.name,
        source_dir=str(source_dir),
        role=role,
        sagemaker_session=session,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version=args.framework_version,
        py_version=args.py_version,
        max_run=int(args.max_run_hours * 3600),
        base_job_name="er-retriever-finetune",
        hyperparameters={
            "model": args.model,
            "epochs": args.epochs,
            "batch-size": args.batch_size,
            "lr": args.lr,
            "max-train-rows": args.max_train_rows,
        },
    )
    estimator.fit({"train": train_s3}, wait=args.wait, logs="All" if args.wait else "None")
    job = estimator.latest_training_job.name
    print(f"\njob: {job}  (SageMaker console > Training > Training jobs)")

    if not args.wait:
        print("re-run with --wait next time to download the model automatically, or fetch it later from:")
        print(f"  {estimator.output_path.rstrip('/')}/{job}/output/model.tar.gz")
        return

    model_s3 = estimator.model_data
    out = Path(args.model_out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"downloading {model_s3} -> {out} ...")
    with tempfile.TemporaryDirectory() as tmp:
        S3Downloader.download(model_s3, tmp, sagemaker_session=session)
        with tarfile.open(Path(tmp) / "model.tar.gz") as tar:
            tar.extractall(out)
    print(f"model ready at {out} (see finetune_config.json for before/after eval accuracy)")


if __name__ == "__main__":
    main()
