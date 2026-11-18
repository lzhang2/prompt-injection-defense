"""Train the classifier with Ray Train and register it in MLflow.

Device rule (used by both functions below): CUDA if the machine has it,
otherwise CPU. The Apple GPU (MPS) is never used inside a Ray worker,
because Ray Train's GPU mode only supports NVIDIA CUDA.
"""
import os
from typing import Optional

import mlflow

from safety.tracking import latest_version


def train_func(config: dict):
    """Runs inside a Ray worker process."""
    import os
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        os.environ["ACCELERATE_USE_CPU"] = "true"

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)
    from ray.train.huggingface.transformers import (RayTrainReportCallback,
                                                    prepare_trainer)
    from safety.classifier.data import load_splits, tokenize

    if device.type == "cpu":
        # After the imports, before TrainingArguments picks a device:
        # keep every library in this worker off the Apple GPU.
        torch.backends.mps.is_available = lambda: False

    ccfg = config["classifier"]
    os.environ["MLFLOW_TRACKING_URI"] = config["mlflow"]["tracking_uri"]
    os.environ["MLFLOW_EXPERIMENT_NAME"] = config["mlflow"]["experiment"]

    tok = AutoTokenizer.from_pretrained(ccfg["base_checkpoint"])
    model = AutoModelForSequenceClassification.from_pretrained(
        ccfg["base_checkpoint"], num_labels=2)
    ds = tokenize(load_splits(ccfg, config.get("extra_train_path")), tok,
                  ccfg["max_length"])

    args = TrainingArguments(
        output_dir="out",
        run_name=config["run_name"],
        learning_rate=ccfg["learning_rate"],
        num_train_epochs=ccfg["epochs"],
        per_device_train_batch_size=ccfg["batch_size"],
        eval_strategy="epoch",   # older transformers: evaluation_strategy
        save_strategy="epoch",   # must match eval so Ray gets a checkpoint per epoch
        report_to="mlflow",
        use_cpu=(device.type == "cpu"),
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                      eval_dataset=ds["test"],
                      data_collator=DataCollatorWithPadding(tok))
    trainer.add_callback(RayTrainReportCallback())

    print("DEVICE CHECK:", args.device, next(model.parameters()).device)
    prepare_trainer(trainer).train()


def run_training(cfg: dict, run_name: str,
                 extra_train_path: Optional[str] = None) -> str:
    """Train, register the model, and return the new MLflow model version."""
    import ray
    import torch
    from ray.train import RunConfig, ScalingConfig
    from ray.train.huggingface.transformers import RayTrainReportCallback
    from ray.train.torch import TorchConfig, TorchTrainer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ray.init(ignore_reinit_error=True)
    loop_config = {"classifier": cfg["classifier"], "mlflow": cfg["mlflow"],
                   "run_name": run_name,
                   "extra_train_path": os.path.abspath(extra_train_path)
                   if extra_train_path else None}
    rcfg = cfg["classifier"]["ray"]

    # GPU mode only when the config asks for it AND this machine has CUDA.
    # On a Mac this is always False, so Ray stays on the CPU.
    use_gpu = bool(rcfg["use_gpu"]) and torch.cuda.is_available()

    result = TorchTrainer(
        train_func,
        train_loop_config=loop_config,
        scaling_config=ScalingConfig(
            num_workers=rcfg["num_workers"], use_gpu=use_gpu,
            resources_per_worker={"CPU": rcfg.get("cpus_per_worker", 1)}),
        torch_config=TorchConfig(backend="nccl" if use_gpu else "gloo"),
        run_config=RunConfig(
            name=run_name,
            storage_path=os.path.abspath(
                os.path.join(cfg["paths"]["artifacts"], "ray_results"))),
    ).fit()

    # Load the final checkpoint that Ray saved.
    with result.checkpoint.as_directory() as ckpt_dir:
        path = os.path.join(ckpt_dir, RayTrainReportCallback.CHECKPOINT_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(path)
    model.config.name_or_path = cfg["classifier"]["base_checkpoint"]
    tok = AutoTokenizer.from_pretrained(cfg["classifier"]["base_checkpoint"])

    # Register it in MLflow and return the new version number.
    name = cfg["mlflow"]["model_name"]
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tok},
        name="model",            # MLflow 2.x: artifact_path="model"
        task="text-classification",
        registered_model_name=name,
    )
    ccfg = cfg["classifier"]
    trained_on = {"dataset": ccfg["dataset"],
                  "dataset_revision": str(ccfg.get("dataset_revision") or "latest"),
                  "base_checkpoint": ccfg["base_checkpoint"],
                  "extra_train_path": os.path.basename(extra_train_path or "") or "none"}
    mlflow.log_params({**trained_on, "learning_rate": ccfg["learning_rate"],
                       "epochs": ccfg["epochs"], "batch_size": ccfg["batch_size"]})
    version = latest_version(name)
    client = mlflow.MlflowClient()       # visible on the version in the registry UI
    for k, v in trained_on.items():
        client.set_model_version_tag(name, version, k, v)
    return version