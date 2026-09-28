"""
FineWeb-Edu loading for the language-model runs.

The trainer expects a Hugging Face dataset saved with ``save_to_disk()`` that
carries an ``input_ids`` column of at least ``seq_length`` tokens per row.  The
paper used the 10BT subset (20BT for the 1B run), tokenized with the Llama-2
SentencePiece tokenizer (vocabulary 32,000) at sequence length 2048, with EOS
between documents.  Neither the corpus nor the tokenizer is redistributed here.

Two layouts are accepted::

    <data_dir>/<dataset_name>/                 # one concatenated dataset
        dataset_info.json  state.json  data-*.arrow

    <data_dir>/<dataset_name>/                 # or sharded
        shard_0000_tokenized/ ...
        shard_0001_tokenized/ ...

The validation split is carved out with a fixed seed (42), independent of the
training seed, so every run in a comparison evaluates on exactly the same held-
out rows.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

DEFAULT_TEST_SIZE = 5.71e-5   # ~280 sequences of a 10BT corpus at seq_length 2048
SPLIT_SEED = 42


class TokenizedDataset(Dataset):
    """Wraps a HF dataset of token ids, truncating rows to `seq_length`.

    Truncation keeps the first `seq_length` tokens, which is unbiased for a
    packed corpus: every row is a contiguous slice of the stream already.
    """

    def __init__(self, dataset, seq_length: Optional[int] = None):
        self.dataset = dataset
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        ids = torch.tensor(self.dataset[idx]["input_ids"], dtype=torch.long)
        if self.seq_length is not None and ids.size(0) > self.seq_length:
            ids = ids[: self.seq_length]
        return {"input_ids": ids}


def _load_raw(data_dir: str, dataset_name: str, num_shards: int = 25):
    import datasets

    root = Path(data_dir) / dataset_name
    if (root / "dataset_info.json").exists():
        return datasets.load_from_disk(str(root))

    shards = []
    for i in range(num_shards):
        shard = root / f"shard_{i:04d}_tokenized"
        if (shard / "dataset_info.json").exists():
            shards.append(datasets.load_from_disk(str(shard)))
    if not shards:
        raise FileNotFoundError(
            f"No dataset found at {root}. Expected either a saved dataset there, "
            f"or shard_XXXX_tokenized/ subdirectories."
        )
    print(f"[data] loaded {len(shards)} shards from {root}")
    return datasets.concatenate_datasets(shards)


def build_datasets(
    data_dir: str,
    dataset_name: str,
    seq_length: int = 2048,
    test_size: float = DEFAULT_TEST_SIZE,
    num_shards: int = 25,
) -> Tuple[TokenizedDataset, TokenizedDataset]:
    """Return ``(train_dataset, eval_dataset)``.

    Raises if the on-disk rows are shorter than `seq_length`; padding would
    change the loss masking, so it is refused rather than done silently.
    """
    full = _load_raw(data_dir, dataset_name, num_shards=num_shards)
    split = full.train_test_split(test_size=test_size, seed=SPLIT_SEED)
    train = TokenizedDataset(split["train"], seq_length)
    val = TokenizedDataset(split["test"], seq_length)

    actual = len(full[0]["input_ids"])
    if actual < seq_length:
        raise ValueError(
            f"Dataset rows hold {actual} tokens but seq_length={seq_length}. "
            f"Use shards with at least {seq_length} tokens per row."
        )
    if actual > seq_length:
        print(f"[data] truncating rows {actual} -> {seq_length} tokens")
    print(f"[data] train={len(train):,} rows, val={len(val):,} rows")
    return train, val


class SyntheticDataset(Dataset):
    """Random token ids, for smoke-testing the pipeline without the corpus."""

    def __init__(self, vocab_size: int = 32000, seq_length: int = 2048, n_rows: int = 512,
                 seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (n_rows, seq_length), generator=g)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        return {"input_ids": self.data[idx]}
