"""Inference adapter for the phase-1 v2 smoothed contact-map head."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from hi_scooby.resources import ResourceRegistry, load_resources


EXPECTED_PAIR_SHAPE = (512, 512, 128)
EXPECTED_CONTEXT_DIM = 14
EXPECTED_MAP_SHAPE = (200, 200)
EXPECTED_PARAMETER_COUNT = 102_271


def _load_model_module(resources: ResourceRegistry) -> ModuleType:
    module_name = "hi_scooby._phase1_model_v2"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    candidates = (
        Path(__file__).resolve().parents[1] / "_legacy_source" / "model_v2.py",
        resources.root / "src" / "scoo-hic" / "model_v2.py",
    )
    model_path = next((path for path in candidates if path.is_file()), None)
    if model_path is None:
        checked = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            "Could not locate the existing phase-1 v2 model source.\n"
            f"Checked:\n{checked}"
        )

    specification = importlib.util.spec_from_file_location(
        module_name,
        model_path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load model module from {model_path}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


class SmoothPredictor:
    """Loaded phase-1 v2 decoder for one AlphaGenome tile at a time."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        model_module: ModuleType,
        checkpoint_path: Path,
        context_ids: tuple[str, ...],
        device: torch.device,
    ) -> None:
        self.model = model
        self.model_module = model_module
        self.checkpoint_path = checkpoint_path
        self.context_ids = context_ids
        self.device = device

    @classmethod
    def load(
        cls,
        resources: ResourceRegistry | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> "SmoothPredictor":
        resources = resources or load_resources()
        checkpoint_path = resources.resolve("phase1_v2_checkpoint")
        model_module = _load_model_module(resources)

        if device is None:
            selected_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            selected_device = torch.device(device)

        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for the smooth head, but PyTorch cannot "
                "access a CUDA device."
            )

        print(
            f"[smooth] Loading phase-1 v2 checkpoint on {selected_device}",
            flush=True,
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint.get("checkpoint_version") != 1:
            raise ValueError(
                "Unsupported phase-1 checkpoint version: "
                f"{checkpoint.get('checkpoint_version')!r}"
            )

        geometry = checkpoint.get("tensor_geometry", {})
        if tuple(geometry.get("pair_embedding_shape", ())) != EXPECTED_PAIR_SHAPE:
            raise ValueError(
                "Checkpoint pair-embedding geometry does not match "
                f"{EXPECTED_PAIR_SHAPE}: {geometry}"
            )
        if tuple(geometry.get("target_map_shape", ())) != EXPECTED_MAP_SHAPE:
            raise ValueError(
                "Checkpoint target-map geometry does not match "
                f"{EXPECTED_MAP_SHAPE}: {geometry}"
            )
        if int(geometry.get("latent_dim", -1)) != EXPECTED_CONTEXT_DIM:
            raise ValueError(
                "Checkpoint RNA latent dimension does not match "
                f"{EXPECTED_CONTEXT_DIM}: {geometry}"
            )

        model = model_module.Phase1ScooHiC()
        model.load_state_dict(checkpoint["model_state"], strict=True)

        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        if parameter_count != EXPECTED_PARAMETER_COUNT:
            raise ValueError(
                f"Unexpected phase-1 parameter count: {parameter_count:,}; "
                f"expected {EXPECTED_PARAMETER_COUNT:,}."
            )

        model.to(selected_device).eval()
        context_ids = tuple(str(value) for value in checkpoint["context_ids"])
        print(
            f"[smooth] Ready: {parameter_count:,} parameters; "
            f"{len(context_ids)} trained contexts",
            flush=True,
        )

        return cls(
            model=model,
            model_module=model_module,
            checkpoint_path=checkpoint_path,
            context_ids=context_ids,
            device=selected_device,
        )

    def predict_tile(
        self,
        pair_embedding: np.ndarray,
        context_embeddings: np.ndarray,
        *,
        input_start: int,
        target_start: int,
    ) -> np.ndarray:
        """Decode one AlphaGenome pair embedding into 5 kb context maps."""
        pair_array = np.asarray(pair_embedding)
        context_array = np.asarray(context_embeddings, dtype=np.float32)

        if pair_array.shape != EXPECTED_PAIR_SHAPE:
            raise ValueError(
                f"pair_embedding must have shape {EXPECTED_PAIR_SHAPE}; "
                f"found {pair_array.shape}"
            )
        if context_array.ndim != 2:
            raise ValueError(
                "context_embeddings must have shape [contexts, 14]; "
                f"found {context_array.shape}"
            )
        if context_array.shape[1] != EXPECTED_CONTEXT_DIM:
            raise ValueError(
                f"context_embeddings must have {EXPECTED_CONTEXT_DIM} columns; "
                f"found {context_array.shape[1]}"
            )
        if context_array.shape[0] == 0:
            raise ValueError("At least one context embedding is required")
        if not np.isfinite(pair_array).all():
            raise ValueError("pair_embedding contains non-finite values")
        if not np.isfinite(context_array).all():
            raise ValueError("context_embeddings contains non-finite values")

        pair_host = np.ascontiguousarray(pair_array)
        context_host = np.ascontiguousarray(context_array)
        if not pair_host.flags.writeable:
            pair_host = pair_host.copy()
        if not context_host.flags.writeable:
            context_host = context_host.copy()

        pair_tensor = torch.from_numpy(
            pair_host
        ).unsqueeze(0).to(self.device)
        context_tensor = torch.from_numpy(
            context_host
        ).to(self.device)
        resample_weights = self.model_module.build_area_overlap_matrix(
            input_start=int(input_start),
            target_start=int(target_start),
            device=self.device,
        ).unsqueeze(0)

        print(
            f"[smooth] Decoding {context_array.shape[0]} context map(s)",
            flush=True,
        )
        with torch.inference_mode():
            prediction = self.model(
                pair_tensor,
                context_tensor,
                resample_weights,
            )[0].detach().cpu().numpy().astype(np.float32, copy=False)

        expected_shape = (
            context_array.shape[0],
            *EXPECTED_MAP_SHAPE,
        )
        if prediction.shape != expected_shape:
            raise RuntimeError(
                f"Smooth prediction has shape {prediction.shape}; "
                f"expected {expected_shape}"
            )
        if not np.isfinite(prediction).all():
            raise RuntimeError("Smooth prediction contains non-finite values")
        if not np.allclose(
            prediction,
            prediction.transpose(0, 2, 1),
            atol=1e-5,
            rtol=0.0,
        ):
            raise RuntimeError("Smooth prediction is not symmetric")

        return prediction
