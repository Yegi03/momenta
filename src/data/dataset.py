"""
src/data/dataset.py
Unified PyTorch Dataset for MOMENTA across all 4 datasets.

Notes:
- Drops rows whose cached image file does not exist on disk.
- Supports optional Fakeddit train-only subsampling via fakeddit_train_frac (seed=42).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────── paths ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # main/
_DATASETS_DIR = _PROJECT_ROOT / "datasets"
_CACHE_DIR = _PROJECT_ROOT / "deployment_cache" / "images"

_DATASET_DIRS: Dict[str, Path] = {
    "mmcovar": _DATASETS_DIR / "MMCoVaR-main",
    "weibo": _DATASETS_DIR / "Weibo-dataset-main",
    "fakeddit": _DATASETS_DIR / "Fakeddit-master" / "data" / "multimodal_only_samples",
    "xfacta": _DATASETS_DIR / "XFacta-main" / "data",
}

# ─────────────────────────── CLIP image transforms ──────────────────────────
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CLIP_TRANSFORM_TRAIN = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ]
)

CLIP_TRANSFORM = T.Compose(
    [
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ]
)

DATASET_IDS: Dict[str, int] = {"fakeddit": 0, "mmcovar": 1, "weibo": 2, "xfacta": 3}


def _build_cache_index():
    stem_idx: dict = {}
    fakeddit_idx: dict = {}
    weibo_hash: dict = {}

    if not _CACHE_DIR.exists():
        return stem_idx, fakeddit_idx, weibo_hash

    for p in _CACHE_DIR.iterdir():
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        s = p.stem
        stem_idx[s] = p
        if s.startswith("fakeddit_"):
            rest = s[len("fakeddit_") :]
            parts = rest.rsplit("_", 1)
            pid = parts[0] if (len(parts) == 2 and len(parts[1]) == 32) else rest
            fakeddit_idx.setdefault(pid, p)
        elif s.startswith("weibo_"):
            if len(s) >= 32:
                weibo_hash[s[-32:]] = p
    return stem_idx, fakeddit_idx, weibo_hash


_CACHE_STEM, _FAKEDDIT_BY_ID, _WEIBO_URL_HASH = _build_cache_index()


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _safe_stem(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^\w\-]", "_", str(s))
    return s[:max_len]


def _parse_timestamp(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass
    return 0.0


def _row_split_mmcovar(news_id) -> str:
    h = int(hashlib.md5(str(news_id).encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


class MOMENTADataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        max_text_len: int = 128,
        transform=None,
        subset_frac: Optional[float] = None,
    ):
        self.dataset_name = dataset_name.lower()
        self.split = split.lower()
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        if transform is not None:
            self.transform = transform
        elif self.split == "train":
            self.transform = CLIP_TRANSFORM_TRAIN
        else:
            self.transform = CLIP_TRANSFORM

        assert self.dataset_name in _DATASET_DIRS
        assert self.split in ("train", "val", "test")

        raw_records = self._parse_records()

        before = len(raw_records)
        self._records = [r for r in raw_records if r["image_path"] is not None and Path(r["image_path"]).is_file()]
        dropped = before - len(self._records)
        if dropped:
            print(f"[{dataset_name}/{split}] Dropped {dropped} rows with missing images ({len(self._records)} kept)")

        if (
            self.dataset_name == "fakeddit"
            and self.split == "train"
            and subset_frac is not None
            and 0.0 < subset_frac < 1.0
        ):
            n_keep = max(1, int(len(self._records) * subset_frac))
            rng = random.Random(42)
            self._records = rng.sample(self._records, n_keep)
            print(f"[fakeddit/train] Subset: {subset_frac*100:.0f}% → {n_keep} samples (seed=42)")

    def _parse_records(self) -> List[dict]:
        parsers = {
            "mmcovar": self._parse_mmcovar,
            "weibo": self._parse_weibo,
            "fakeddit": self._parse_fakeddit,
            "xfacta": self._parse_xfacta,
        }
        return parsers[self.dataset_name]()

    def _parse_mmcovar(self) -> List[dict]:
        import pandas as pd

        d = _DATASET_DIRS["mmcovar"]
        df = pd.read_csv(d / "MMCoVaR_News_Dataset.csv")

        before = len(df)
        df = df[df["image"].notna() & (df["image"].str.strip() != "")].copy()
        if len(df) < before:
            print(f"[mmcovar] Dropped {before - len(df)} rows with null image column")

        records = []
        for _, row in df.iterrows():
            if _row_split_mmcovar(row["news_id"]) != self.split:
                continue

            url = str(row["image"])
            img_stem = f"mmcovar_{row['news_id']}_{_url_hash(url)}"
            img_path = _CACHE_STEM.get(img_stem)

            reliability = int(row.get("reliability", 1))
            label = 0 if reliability == 1 else 1
            y_match = 1 if reliability == 1 else 0

            text = " ".join(
                filter(
                    None,
                    [
                        str(row.get("title", "") or ""),
                        str(row.get("body_text", "") or "")[:512],
                    ],
                )
            ).strip() or "[NO TEXT]"

            records.append(
                {
                    "text": text,
                    "image_path": img_path,
                    "label": label,
                    "y_match": y_match,
                    "timestamp": _parse_timestamp(row.get("publish_date")),
                    "post_id": str(row["news_id"]),
                    "dataset": "mmcovar",
                }
            )
        return records

    def _parse_weibo(self) -> List[dict]:
        file_split = "test" if self.split == "val" else self.split
        d = _DATASET_DIRS["weibo"]
        label_map = {"rumor": 1, "nonrumor": 0}
        ymatch_map = {"rumor": 0, "nonrumor": 1}
        records: List[dict] = []

        for label_str, label_int in label_map.items():
            filepath = d / f"{file_split}_{label_str}.txt"
            if not filepath.exists():
                continue
            with open(filepath, encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
            i = 0
            while i + 2 < len(lines):
                meta_line = lines[i].strip()
                img_line = lines[i + 1].strip()
                text_line = lines[i + 2].strip()
                i += 3
                if "|" not in meta_line:
                    i -= 2
                    continue
                parts = meta_line.lstrip("\ufeff").split("|")
                tweet_id = parts[0].strip()
                date_str = parts[4].strip() if len(parts) > 4 else ""
                tid = _safe_stem(tweet_id, 40)
                img_path = self._find_weibo_image(img_line, tid)

                records.append(
                    {
                        "text": text_line or "[NO TEXT]",
                        "image_path": img_path,
                        "label": label_int,
                        "y_match": ymatch_map[label_str],
                        "timestamp": _parse_timestamp(date_str),
                        "post_id": tweet_id,
                        "dataset": "weibo",
                    }
                )
        return records

    def _find_weibo_image(self, img_line: str, tid: str) -> Optional[Path]:
        for url in img_line.split("|"):
            url = url.strip()
            if not url.startswith("http"):
                continue
            h = _url_hash(url)
            if f"weibo_{tid}_{h}" in _CACHE_STEM:
                return _CACHE_STEM[f"weibo_{tid}_{h}"]
            if h in _WEIBO_URL_HASH:
                return _WEIBO_URL_HASH[h]
        return None

    def _parse_fakeddit(self) -> List[dict]:
        import pandas as pd

        d = _DATASET_DIRS["fakeddit"]
        split_files = {
            "train": "multimodal_train.tsv",
            "val": "multimodal_validate.tsv",
            "test": "multimodal_test_public.tsv",
        }
        df = pd.read_csv(d / split_files[self.split], sep="\t")

        records = []
        for _, row in df.iterrows():
            img_path = self._find_fakeddit_image(str(row["id"]), str(row.get("image_url", "")))
            label = int(row.get("2_way_label", 0))
            y_match = 1 if int(row.get("6_way_label", 1)) == 0 else 0
            text = str(row.get("clean_title") or row.get("title") or "").strip() or "[NO TEXT]"
            records.append(
                {
                    "text": text,
                    "image_path": img_path,
                    "label": label,
                    "y_match": y_match,
                    "timestamp": _parse_timestamp(row.get("created_utc")),
                    "post_id": str(row["id"]),
                    "dataset": "fakeddit",
                }
            )
        return records

    def _find_fakeddit_image(self, post_id: str, image_url: str) -> Optional[Path]:
        if image_url.startswith("http"):
            stem = f"fakeddit_{post_id}_{_url_hash(image_url)}"
            if stem in _CACHE_STEM:
                return _CACHE_STEM[stem]
        plain = f"fakeddit_{post_id}"
        if plain in _CACHE_STEM:
            return _CACHE_STEM[plain]
        return _FAKEDDIT_BY_ID.get(post_id)

    def _parse_xfacta(self) -> List[dict]:
        xf = _DATASET_DIRS["xfacta"]
        if self.split in ("val", "test"):
            fname = "dev.json" if self.split == "val" else "test.json"
            with open(xf / fname, encoding="utf-8") as f:
                items = json.load(f)
            return self._xfacta_from_flat_list(items, xf)

        records: List[dict] = []
        for sample_type in ("fake_sample", "real_sample"):
            sample_dir = xf / sample_type
            for i in range(1, 13):
                bf = sample_dir / f"batch{i}.json"
                if not bf.exists():
                    continue
                with open(bf, encoding="utf-8") as f:
                    batch = json.load(f)
                for item in batch:
                    tweet = item.get("ooc_tweet", {})
                    meta = tweet.get("metadata", {})
                    label_val = meta.get("label")
                    if label_val is None:
                        continue
                    text = tweet.get("text", "").strip() or "[NO TEXT]"
                    images = tweet.get("images", [])
                    date = meta.get("date_posted", "")
                    tid = str(item.get("tweet_id", ""))
                    img_path = self._resolve_xfacta_image(images, sample_dir)
                    lbl = 0 if label_val else 1
                    y_match = 1 if label_val else 0
                    records.append(
                        {
                            "text": text,
                            "image_path": img_path,
                            "label": lbl,
                            "y_match": y_match,
                            "timestamp": _parse_timestamp(date),
                            "post_id": tid,
                            "dataset": "xfacta",
                        }
                    )
        return records

    def _xfacta_from_flat_list(self, items: list, xf: Path) -> List[dict]:
        records = []
        for i, item in enumerate(items):
            label_val = item.get("label")
            if label_val is None:
                continue
            text = item.get("text", "").strip() or "[NO TEXT]"
            images = item.get("images", [])
            img_path = self._resolve_xfacta_image_abs(images, xf)
            lbl = 0 if label_val else 1
            y_match = 1 if label_val else 0
            records.append(
                {
                    "text": text,
                    "image_path": img_path,
                    "label": lbl,
                    "y_match": y_match,
                    "timestamp": 0.0,
                    "post_id": f"xfacta_{self.split}_{i}",
                    "dataset": "xfacta",
                }
            )
        return records

    def _resolve_xfacta_image(self, images: list, sample_dir: Path) -> Optional[Path]:
        for rel in images:
            rel = str(rel).lstrip("./")
            p = sample_dir / rel
            if p.exists():
                return p
        return None

    def _resolve_xfacta_image_abs(self, images: list, xf: Path) -> Optional[Path]:
        server_prefix = "/projects/vig/hzy/XFacta/"
        for abs_path in images:
            s = str(abs_path)
            if s.startswith(server_prefix):
                rel = s[len(server_prefix) :]
            elif s.startswith("/"):
                idx = s.find("XFacta/")
                rel = s[idx + len("XFacta/") :] if idx != -1 else s.lstrip("/")
            else:
                rel = s.lstrip("./")
            p = xf / rel
            if p.exists():
                return p
        return None

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self._records[idx]
        enc = self.tokenizer(
            rec["text"],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        img = Image.open(rec["image_path"]).convert("RGB")
        image_tensor = self.transform(img)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": image_tensor,
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "y_match": torch.tensor(rec["y_match"], dtype=torch.long),
            "timestamp": torch.tensor(rec["timestamp"], dtype=torch.float32),
            "dataset_id": torch.tensor(DATASET_IDS.get(rec["dataset"], 0), dtype=torch.long),
            "post_id": rec["post_id"],
            "dataset": rec["dataset"],
        }


def build_dataloaders(
    dataset_names: List[str],
    tokenizer,
    batch_size: int = 32,
    max_text_len: int = 128,
    num_workers: int = 4,
    fakeddit_train_frac: Optional[float] = None,
) -> Dict[str, Dict[str, DataLoader]]:
    def _collate(batch):
        keys_tensor = ["input_ids", "attention_mask", "pixel_values", "label", "y_match", "timestamp", "dataset_id"]
        keys_str = ["post_id", "dataset"]
        out = {k: torch.stack([b[k] for b in batch]) for k in keys_tensor}
        out.update({k: [b[k] for b in batch] for k in keys_str})
        return out

    loaders: Dict[str, Dict[str, DataLoader]] = {}
    for name in dataset_names:
        loaders[name] = {}
        for split in ("train", "val", "test"):
            subset_frac = fakeddit_train_frac if (name == "fakeddit" and split == "train") else None
            ds = MOMENTADataset(
                dataset_name=name,
                split=split,
                tokenizer=tokenizer,
                max_text_len=max_text_len,
                subset_frac=subset_frac,
            )
            loaders[name][split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                collate_fn=_collate,
                pin_memory=True,
            )
    return loaders

