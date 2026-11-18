"""Load a registered classifier version from MLflow and score texts."""
from typing import List, Sequence


def model_uri(cfg: dict, ref: str) -> str:
    name, ref = cfg["mlflow"]["model_name"], str(ref)
    return f"models:/{name}@{ref[1:]}" if ref.startswith("@") else f"models:/{name}/{ref}"


def resolve_version(cfg: dict, ref: str) -> str:
    """Turn "@champion" into the version number it points to right now."""
    ref = str(ref)
    if not ref.startswith("@"):
        return ref
    import mlflow

    mv = mlflow.MlflowClient().get_model_version_by_alias(cfg["mlflow"]["model_name"], ref[1:])
    return str(mv.version)


class InjectionClassifier:
    def __init__(self, model, tok, max_length: int = 512, batch_size: int = 16):
        import torch

        self._torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model = model.to(self.device).eval()
        self.tok, self.max_length, self.batch_size = tok, max_length, batch_size

    @classmethod
    def from_registry(cls, cfg: dict, version: str) -> "InjectionClassifier":
        """`version` is a number ("2") or an alias with a leading @ ("@champion")."""
        import mlflow

        uri = model_uri(cfg, version)
        parts = mlflow.transformers.load_model(uri, return_type="components")
        return cls(parts["model"], parts["tokenizer"],
                   max_length=cfg["classifier"]["max_length"])

    def score(self, texts: Sequence[str]) -> List[float]:
        """Probability of 'injection' (label 1) for each text."""
        torch, out = self._torch, []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                probs = self.model(**enc).logits.softmax(-1)[:, 1]
            out.extend(probs.cpu().tolist())
        return out
