"""On-the-fly mouse AlphaGenome pair-embedding extraction."""

from __future__ import annotations

from pathlib import Path

from alphagenome.models import dna_model as public_dna_model
from alphagenome_research.io import fasta as fasta_lib
from alphagenome_research.model import dna_model as research_dna_model
from alphagenome_research.model import model as model_lib
from alphagenome.data import genome
import haiku as hk
import huggingface_hub
import jax
import jax.numpy as jnp
import jmp
import numpy as np


MODEL_VERSION = "all_folds"
HUGGINGFACE_REPO = "google/alphagenome-all-folds"
HUGGINGFACE_REVISION = "a8f293a76ee73d5b57f3bf2ae146510589fcf187"
SEQUENCE_LENGTH_BP = 1_048_576
PAIR_BINS = 512
PAIR_CHANNELS = 128


class AlphaGenomePairEmbedder:
    """Restored AlphaGenome model exposing its mouse pair embedding."""

    def __init__(
        self,
        restored_model: research_dna_model.AlphaGenomeModel,
        fasta_path: str | Path,
    ) -> None:
        required_fields = (
            "_params",
            "_state",
            "_metadata",
            "_device_context",
            "_one_hot_encoder",
        )
        missing = [
            field
            for field in required_fields
            if not hasattr(restored_model, field)
        ]
        if missing:
            raise RuntimeError(
                "Pinned AlphaGenome wrapper is missing required fields: "
                f"{missing}"
            )

        self._model = restored_model
        self._fasta = fasta_lib.FastaExtractor(
            Path(fasta_path).expanduser().resolve()
        )
        self._first_forward = True

        metadata = restored_model._metadata
        policy = jmp.get_policy(
            "params=float32,compute=bfloat16,output=bfloat16"
        )

        @hk.transform_with_state
        def forward_embeddings(
            dna_sequence: jax.Array,
            organism_index: jax.Array,
        ):
            with hk.mixed_precision.push_policy(
                model_lib.AlphaGenome,
                policy,
            ):
                _, embeddings = model_lib.AlphaGenome(metadata)(
                    dna_sequence,
                    organism_index,
                )
            return embeddings

        self._apply_embeddings = jax.jit(forward_embeddings.apply)

    def get_sequence(
        self,
        chrom: str,
        start: int,
        end: int,
    ) -> str:
        """Extract one exact AlphaGenome window from mm10."""

        if end - start != SEQUENCE_LENGTH_BP:
            raise ValueError(
                "AlphaGenome interval must span exactly "
                f"{SEQUENCE_LENGTH_BP:,} bp; received {end - start:,}"
            )

        interval = genome.Interval(
            chromosome=chrom,
            start=int(start),
            end=int(end),
        )
        sequence = self._fasta.extract(interval)
        if len(sequence) != SEQUENCE_LENGTH_BP:
            raise RuntimeError(
                f"FASTA returned {len(sequence):,} bp for "
                f"{chrom}:{start}-{end}"
            )
        return sequence

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Run one sequence through AlphaGenome and return float32 pair data."""

        if len(sequence) != SEQUENCE_LENGTH_BP:
            raise ValueError(
                f"Expected {SEQUENCE_LENGTH_BP:,} bp; "
                f"received {len(sequence):,}"
            )

        if self._first_forward:
            print(
                "[AlphaGenome] Compiling the first pair-embedding forward pass",
                flush=True,
            )

        organism = public_dna_model.Organism.MUS_MUSCULUS
        organism_index_value = research_dna_model.convert_to_organism_index(
            organism
        )
        if organism_index_value != 1:
            raise RuntimeError(
                "Pinned AlphaGenome source no longer maps mouse to index 1"
            )

        with self._model._device_context as device:
            one_hot = self._model._one_hot_encoder.encode(sequence)
            one_hot = jax.device_put(
                np.asarray(one_hot, dtype=np.float32)[None, ...],
                device,
            )
            organism_index = jax.device_put(
                np.asarray([organism_index_value], dtype=np.int32),
                device,
            )

            embeddings, _ = self._apply_embeddings(
                self._model._params,
                self._model._state,
                None,
                one_hot,
                organism_index,
            )
            if embeddings.embeddings_pair is None:
                raise RuntimeError(
                    "AlphaGenome forward returned no pair embedding"
                )

            pair = np.asarray(
                jax.device_get(
                    embeddings.embeddings_pair.astype(jnp.float32)
                )
            )

        expected_shape = (1, PAIR_BINS, PAIR_BINS, PAIR_CHANNELS)
        if pair.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected AlphaGenome pair shape {pair.shape}; "
                f"expected {expected_shape}"
            )
        if not np.isfinite(pair).all():
            raise RuntimeError(
                "AlphaGenome pair embedding contains non-finite values"
            )

        if self._first_forward:
            print("[AlphaGenome] First forward pass compiled", flush=True)
            self._first_forward = False

        return pair[0]

    def embed_interval(
        self,
        chrom: str,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Extract mm10 sequence and return its pair embedding."""

        return self.embed_sequence(self.get_sequence(chrom, start, end))


def load_pair_embedder(
    fasta_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
) -> AlphaGenomePairEmbedder:
    """Load the pinned all-folds checkpoint and mm10 FASTA."""

    if checkpoint_path is None:
        print(
            "[AlphaGenome] Resolving pinned all_folds checkpoint",
            flush=True,
        )
        checkpoint_path = huggingface_hub.snapshot_download(
            repo_id=HUGGINGFACE_REPO,
            revision=HUGGINGFACE_REVISION,
        )
    else:
        checkpoint_path = str(
            Path(checkpoint_path).expanduser().resolve()
        )

    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError(
            "AlphaGenome inference requires a visible JAX GPU"
        )

    print(
        f"[AlphaGenome] Restoring checkpoint on {devices[0]}",
        flush=True,
    )
    restored_model = research_dna_model.create(
        checkpoint_path,
        device=devices[0],
    )
    print("[AlphaGenome] Checkpoint restored", flush=True)

    return AlphaGenomePairEmbedder(
        restored_model=restored_model,
        fasta_path=fasta_path,
    )