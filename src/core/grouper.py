from datetime import datetime
import math
import gzip
import torch
import os
import json
import traceback
from time import time, sleep
from typing import Optional, List, Tuple, Dict, Sequence, Any, Iterable
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from sklearn.cluster import KMeans
import torch.nn.functional as F
from src.config.arguments import BAITArguments
from openai import OpenAI
from src.utils.constants import JUDGE_SYSTEM_PROMPT
from src.config.arguments import ModelArguments, DataArguments, ScanArguments
from src.utils.helpers import extract_tag
from openai import APIError, RateLimitError, APIConnectionError
from dataclasses import dataclass
from loguru import logger
from src.models.model import build_model, parse_model_args
from src.data.dataset import build_data_module
import sys
import hashlib
from collections import defaultdict
import random



class Grouper:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        logger):
        logger.info("Start Grouping...")
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.logger = logger

    def build_panel(self, L: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make a small, reusable panel of contexts to probe token behavior.

        Returns:
            panel_inputs: [R, L] LongTensor
            panel_masks:  [R, L] LongTensor in {0,1}
        """
        self.logger.info("Building panel...")
        panel_texts = [
            "I read the article about climate policy and",
            "Please list the steps to connect to the Wi-Fi.",
            "According to the report, the results show",
            "He said, in other words,",
            "In Python, a function can return",
            "The value after the loop is",
            "Here are three reasons why",
            "The new research indicates that",
            "First, consider the following example:",
            "Note: when the file is missing, the program",
            "As a reminder, the meeting begins at",
            "Question: what is the main point of",
            "If the input is empty, then",
            "The equation f(x) = x + 1 implies that",
            "She opened the door and said,",
            "To reproduce the results, follow these steps:",
            "After saving the file, restart the application and",
            "It was cold; however, the device still",
            "Open the settings (Preferences) and then select",
            "For reference, the documentation mentions that",
        ]

        enc = self.tokenizer(
            list(panel_texts),
            padding="max_length",
            truncation=True,
            max_length=L,
            return_tensors="pt",
            add_special_tokens=True,
        )
        panel_inputs = enc["input_ids"]            # [R, L]
        panel_masks  = enc["attention_mask"]       # [R, L]
        return panel_inputs, panel_masks
    
    def get_space_prefix(self) -> str:
        no_space = self.tokenizer.tokenize("the")
        with_space = self.tokenizer.tokenize(" the")

        if len(with_space) == 1 and with_space[0] != no_space[0]:
            diff = with_space[0].replace(no_space[0], "")
            return diff
        return ""

    def build_landmarks(self):
        # base list without prefix
        self.logger.info("Building landmarks...")
        landmarks = [
            "the","a","an","and","or","of","to","in","on","for",
            "he","she","they","we","I",
            "is","was","has","do",
            ".",",",":",";","?","!","(",")","\"","'",
            "0","1","2"
        ]
        prefix = self.get_space_prefix()

        # prepend prefix only to word-like tokens (not punctuation/numbers)
        word_like = set([
            "the","a","an","and","or","of","to","in","on","for",
            "he","she","they","we","I",
            "is","was","has","do"
        ])
        landmarks_with_prefix = [
            prefix + tok if tok in word_like else tok
            for tok in landmarks
        ]

        return landmarks_with_prefix
    
    def _tensor_shape(self, t: torch.Tensor) -> Tuple[int, ...]:
        return tuple(int(x) for x in t.shape)

    def _resolve_landmark_ids(self, landmarks: Optional[Sequence[Any]]) -> List[int]:
        """Resolve landmark specs (strings or ids) to unique, sorted token ids."""
        if not landmarks:
            return []
        ids: List[int] = []
        for lm in landmarks:
            if isinstance(lm, int):
                ids.append(lm)
            else:
                tid = self.tokenizer.convert_tokens_to_ids(lm)
                if tid is None or tid < 0:
                    enc = self.tokenizer.encode(str(lm), add_special_tokens=False)
                    if enc:
                        tid = enc[0]
                    else:
                        continue
                ids.append(tid)
        return sorted(set(ids))

    def compute_groups_config_hash(
        self,
        panel_inputs: torch.Tensor,
        panel_masks: torch.Tensor,
        positions: Sequence[int],
        temps: Sequence[float],
        landmarks: Optional[Sequence[Any]],
        n_groups: int,
        topk_mass: int,
        chunk_tokens: int,
        compress_dim: int,
    ) -> str:
        """Build a stable hash for the grouping configuration."""
        cfg = {
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", str(type(self.tokenizer))),
            "vocab_size": int(self.tokenizer.vocab_size),
            "bos": getattr(self.tokenizer, "bos_token_id", None),
            "eos": getattr(self.tokenizer, "eos_token_id", None),
            "pad": getattr(self.tokenizer, "pad_token_id", None),
            "panel_inputs_shape": self._tensor_shape(panel_inputs),
            "panel_masks_shape": self._tensor_shape(panel_masks),
            "positions": [p for p in positions],
            "temps": [t for t in temps],
            "landmarks_ids": self._resolve_landmark_ids(landmarks),
            "n_groups": n_groups,
            "topk_mass": topk_mass,
            "chunk_tokens": chunk_tokens,
            "compress_dim": compress_dim
        }

        blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


    # ------------------------------------------------------------
    # 1) Save / Load (gzip JSON). We store labels as a list of len V.
    # ------------------------------------------------------------
    def save_token_groups(
        self,
        path: str,
        id2group: Dict[int, int],
        config_hash: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save mapping and metadata.
        File format (gzip JSON):
        {
            "meta": {
            "created_at": ISO8601,
            "tokenizer_name": "...",
            "vocab_size": V,
            "config_hash": "...",
            ...extra_meta
            },
            "labels": [g0, g1, ..., g(V-1)]  # group for token id == index
        }
        """
        self.logger.info("Saving token groups...")
        V = self.tokenizer.vocab_size
        labels = [-1] * V
        for tid, gid in id2group.items():
            if 0 <= int(tid) < V:
                labels[int(tid)] = int(gid)

        meta = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", str(type(self.tokenizer))),
            "vocab_size": V,
            "config_hash": config_hash,
        }
        if extra_meta:
            meta.update(extra_meta)

        payload = {"meta": meta, "labels": labels}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)

    def load_token_groups(self, path: str) -> Tuple[Dict[int, int], Dict[str, Any]]:
        """
        Load mapping and metadata. Returns (id2group, meta).
        Raises FileNotFoundError if path missing.
        """
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        labels: List[int] = payload["labels"]
        id2group = {i: int(g) for i, g in enumerate(labels)}
        meta = payload.get("meta", {})
        self.logger.info(f"Loaded token groups from {path}")
        return id2group, meta


    # ------------------------------------------------------------
    # 2) Cache-first wrapper around the sklearn builder
    # ------------------------------------------------------------
    def group(
        self,
        cache_path: str = None,
        positions: Sequence[int] = (0, 1, 2),
        temps: Sequence[float] = (0.7, 1.0, 1.3),
        n_groups: int = 128,
        topk_mass: int = 5,
        chunk_tokens: int = 512,
        compress_dim: int = 24
    ) -> Dict[int, int]:
        """
        Try to load cached groups. If missing/mismatched, build and save.
        Returns: dict {token_id: group_id} (all vocab ids present; specials → -1)
        """
        # Load panel inputs and masks
        panel_inputs, panel_masks = self.build_panel()
        # Load landmarks
        landmarks = self.build_landmarks()
        # Compute expected config hash
        expected_hash = self.compute_groups_config_hash(
            panel_inputs=panel_inputs,
            panel_masks=panel_masks,
            positions=positions,
            temps=temps,
            landmarks=landmarks,
            n_groups=n_groups,
            topk_mass=topk_mass,
            chunk_tokens=chunk_tokens,
            compress_dim=compress_dim
        )

        # Try load
        try:
            id2group, meta = self.load_token_groups(cache_path)
            same_vocab = meta.get("vocab_size", -1) == self.tokenizer.vocab_size
            same_hash = meta.get("config_hash") == expected_hash
            if same_vocab and same_hash:
                self.logger.info("Loading token groups from cache...")
                return id2group
            # else, fall through to rebuild
        except FileNotFoundError:
            pass

        self.logger.info("Building token groups (this may take a while)...")

        # Build fresh
        id2group = self.build_token_groups(
            panel_inputs=panel_inputs,
            panel_masks=panel_masks,
            positions=positions,
            temps=temps,
            landmarks=landmarks,
            n_groups=n_groups,
            topk_mass=topk_mass,
            chunk_tokens=chunk_tokens,
            compress_dim=compress_dim,
        )

        # Save
        self.logger.info("Saving token groups to cache...")
        self.save_token_groups(
            cache_path,
            id2group,
            tokenizer=self.tokenizer,
            config_hash=expected_hash,
            extra_meta={
                "positions": [int(p) for p in positions],
                "temps": [float(t) for t in temps],
                "landmarks_ids": self._resolve_landmark_ids(landmarks),
                "n_groups": int(n_groups),
                "topk_mass": int(topk_mass),
                "chunk_tokens": int(chunk_tokens),
                "compress_dim": int(compress_dim),
                "builder": "sklearn",
            },
        )
        return id2group
    
    def build_token_groups(
        self,
        panel_inputs: torch.Tensor,                 # [R, L]
        panel_masks: torch.Tensor,                  # [R, L]
        positions,     # -1 => L-1
        temps,
        landmarks,  # token strings or ids; optional
        n_groups: int,
        topk_mass: int,
        chunk_tokens: int,                    # how many token ids per GPU pass
        compress_dim: int,                     # random projection output dim (<= features => no-op)
    ) -> Dict[int, int]:
        """
        Target-independent, black-box Stage-A grouping:
        • builds per-(token, position) behavioral fingerprints (entropy/top-k/optional landmarks)
        • per-position z-score normalization
        • combines across positions as [mean || std]
        • random-projection compression
        • sklearn KMeans clustering → groups
        • returns {token_id: group_id} for all vocab ids (specials mapped to -1)
        """
        device = self.device
        panel_inputs = panel_inputs.to(device)
        panel_masks  = panel_masks.to(device)
        R, L = panel_inputs.shape

        # Resolve positions (turn -1 into L-1, validate)
        pos_list: List[int] = []
        for p in positions:
            pos = L - 1 if p == -1 else int(p)
            if not (0 <= pos < L):
                raise ValueError(f"Position {p} resolved to {pos} is out of range for L={L}.")
            pos_list.append(pos)

        # token ids (default: all except specials)
        specials = {
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        }
        allowed = [i for i in range(self.tokenizer.vocab_size) if i not in specials]
        token_ids = torch.tensor(allowed, dtype=torch.long, device=device)
        N = token_ids.numel()

        # Landmarks (optional) → ids
        landmark_ids: Optional[torch.Tensor] = None
        if landmarks:
            lm_ids: List[int] = []
            for lm in landmarks:
                tid = self.tokenizer.convert_tokens_to_ids(lm)
                if tid is None or tid < 0:
                    continue
                lm_ids.append(tid)
            if lm_ids:
                lm_ids = sorted(set(lm_ids))
                landmark_ids = torch.tensor(lm_ids, dtype=torch.long, device=device)

        # ---------- helpers ----------
        def features_for_chunk_at_pos(z_chunk: torch.Tensor, pos: int) -> torch.Tensor:
            """
            Compute per-temp, per-token features for one position.
            Returns: [Nc, len(temps)*(4 + 2*K)]  where K = len(landmarks) or 0
                For each temperature:
                H_mean, H_std, topK_mean, topK_std, (optional) LP_mean[K], LP_std[K]
            """
            Nc = z_chunk.numel()
            # base panels repeated per token id (do once, reuse across temps)
            base_inp = panel_inputs.repeat(Nc, 1)  # [Nc*R, L]
            base_msk = panel_masks.repeat(Nc, 1)   # [Nc*R, L]
            base_inp[:, pos] = z_chunk.repeat_interleave(R)
            base_msk[:, pos] = 1

            per_temp_blocks: List[torch.Tensor] = []
            for temp in temps:
                probs = self._simple_generate(base_inp, base_msk, temp)        # [Nc*R, V]
                p = probs.clamp_min(1e-12)

                # Entropy per row: sum(-p log p)
                H = (-p * p.log()).sum(dim=1)                                      # [Nc*R]
                # Top-k mass per row
                top_mass = torch.topk(probs, k=topk_mass, dim=1).values.sum(dim=1)  # [Nc*R]

                # Aggregate over R contexts
                H_mean  = H.view(Nc, R).mean(dim=1)                                 # [Nc]
                H_std   = H.view(Nc, R).std(dim=1)                                  # [Nc]
                Tm_mean = top_mass.view(Nc, R).mean(dim=1)                          # [Nc]
                Tm_std  = top_mass.view(Nc, R).std(dim=1)                           # [Nc]

                if landmark_ids is not None and landmark_ids.numel() > 0:
                    LP = probs[:, landmark_ids]                                     # [Nc*R, K]
                    K = LP.size(1)
                    LP_mean = LP.view(Nc, R, K).mean(dim=1)                         # [Nc, K]
                    LP_std  = LP.view(Nc, R, K).std(dim=1)                          # [Nc, K]
                    block = torch.cat(
                        [H_mean[:, None], H_std[:, None], Tm_mean[:, None], Tm_std[:, None], LP_mean, LP_std],
                        dim=1
                    )                                                               # [Nc, 4+2K]
                else:
                    block = torch.stack([H_mean, H_std, Tm_mean, Tm_std], dim=1)    # [Nc, 4]

                per_temp_blocks.append(block)

            return torch.cat(per_temp_blocks, dim=1)                                 # [Nc, Dp]

        def zscore_per_position(X: torch.Tensor) -> torch.Tensor:
            """Z-score normalize features per position across tokens: X -> (X-mu)/sigma."""
            mu = X.mean(dim=0, keepdim=True)
            sd = X.std(dim=0, keepdim=True) + 1e-8
            return (X - mu) / sd

        def random_project(X: torch.Tensor, out_dim: int) -> torch.Tensor:
            """
            Dense Gaussian random projection to out_dim, then L2 normalize.
            If out_dim >= D, returns L2-normalized X.
            """
            X = X.to(dtype=torch.float32)
            N, D = X.shape
            if out_dim is None or out_dim <= 0 or out_dim >= D:
                Y = F.normalize(X, p=2, dim=1)
                return Y
            # Gaussian RP: W ~ N(0, 1/sqrt(D))
            W = torch.randn((D, out_dim), device=X.device, dtype=X.dtype) / math.sqrt(D)
            Y = X @ W
            return F.normalize(Y, p=2, dim=1)

        # -------------------------------------------------------
        # Build per-position feature matrices for all ids
        # -------------------------------------------------------
        per_pos_feats: List[torch.Tensor] = []
        for pos in pos_list:
            blocks: List[torch.Tensor] = []
            for start in range(0, N, chunk_tokens):
                end = min(start + chunk_tokens, N)
                z_chunk = token_ids[start:end]  # [Nc]
                feats = features_for_chunk_at_pos(z_chunk, pos)  # [Nc, Dp]
                blocks.append(feats)
            Xp = torch.cat(blocks, dim=0)       # [N, Dp] features for this position
            Xp = zscore_per_position(Xp)        # remove slot bias
            per_pos_feats.append(Xp)

        # -------------------------------------------------------
        # Combine across positions: [mean || std] (pos-robustness)
        # -------------------------------------------------------
        Xstack = torch.stack(per_pos_feats, dim=0)  # [P, N, Dp]
        X_mean = Xstack.mean(dim=0)                 # [N, Dp]
        X_std  = Xstack.std(dim=0)                  # [N, Dp]
        X_all  = torch.cat([X_mean, X_std], dim=1)  # [N, 2*Dp]

        # -------------------------------------------------------
        # Compress (RP) + sklearn KMeans
        # -------------------------------------------------------
        Xc = random_project(X_all, out_dim=compress_dim)        # [N, d]
        Xc_np = Xc.detach().cpu().numpy()                       # sklearn expects numpy

        self.logger.info("Clustering token groups...")

        km = KMeans(n_clusters=n_groups, n_init="auto")  # clean sklearn API; no random_state
        labels_np = km.fit_predict(Xc_np)

        # Map ids
        id2group = {int(tid.item()): int(lbl) for tid, lbl in zip(token_ids, torch.from_numpy(labels_np))}

        # Add entries for the rest of the vocab (e.g., specials) as -1
        for vid in range(self.tokenizer.vocab_size):
            if vid not in id2group:
                id2group[vid] = -1

        return id2group