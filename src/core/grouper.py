import os
import torch
import json
from typing import Dict
from transformers import PreTrainedTokenizer
import hashlib
import numpy as np
import string
import unicodedata


class Grouper:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        logger):
        logger.info("Start Grouping...")
        self.tokenizer = tokenizer
        self.logger = logger


    # ------------------------------------------------------------
    # 1) Save. We store labels as a list of len V.
    # ------------------------------------------------------------

    def write_id2group(self, data: Dict[int, int], path: str) -> None:
        """Write dict[int, int] to a JSONL file, ensuring directories exist."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for k, v in data.items():
                record = {"id": k, "group": v}
                f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------
    # 2) Cache-first wrapper around the sklearn builder
    # ------------------------------------------------------------
    @torch.no_grad()
    def group(
        self,
        cache_path: str = None,
        n_groups: int = 128,
        compress_dim: int = 24
    ) -> Dict[int, int]:
        
        self.logger.info("Building token groups...")
        id2group = self.build_token_groups(
            n_groups=n_groups,
            compress_dim=compress_dim
        )

        self.logger.info("Token grouping complete.")
        if cache_path:
            self.write_id2group(id2group, cache_path)

        return id2group


    def build_token_groups(
        self,
        n_groups: int,
        compress_dim: int,
    ) -> Dict[int, int]:
        """
        Group tokens by their string/byte representation only (no model).
        We build hashed byte n-gram features (+ small lexical flags) for each token
        and run MiniBatchKMeans on CPU.

        Returns: dict {token_id: group_id}, specials → -1
        """
        try:
            # Much faster for 32k tokens
            from sklearn.cluster import MiniBatchKMeans as _KMeans
        except Exception:
            # Fall back to regular KMeans if MiniBatchKMeans unavailable
            from sklearn.cluster import KMeans as _KMeans

        V = int(self.tokenizer.vocab_size)
        self.logger.info(f"Tokenizer-only grouping over vocab size={V}")

        # --- Configure features ---
        hash_dim = int(compress_dim) if (compress_dim and compress_dim > 0) else 512
        ngram_min, ngram_max = 2, 5
        rng_seed = 0  # stable features across runs
        flags_dim = 8  # number of lexical flags we append to hashed feats
        feat_dim = hash_dim + flags_dim

        # --- Build id -> token string table in id order ---
        tok_to_id = self.tokenizer.get_vocab()
        id_to_tok = [None] * V
        for tok, tid in tok_to_id.items():
            if 0 <= int(tid) < V:
                id_to_tok[int(tid)] = tok
        # Some tokenizers might have gaps; fill with placeholder
        for i in range(V):
            if id_to_tok[i] is None:
                id_to_tok[i] = ""

        # --- Helpers for feature extraction ---
        def _hash_ngram(b: bytes) -> int:
            # Stable 64-bit hash -> index in [0, hash_dim)
            h = hashlib.blake2b(b, digest_size=8, person=b"grouper", key=rng_seed.to_bytes(4, "little"))
            return int.from_bytes(h.digest(), "little") % hash_dim

        boundary_prefixes = ("Ġ", "▁", " ")  # GPT-2/NeoX, SentencePiece, plain-space

        punct_set = set(string.punctuation)

        def token_features(tok: str) -> np.ndarray:
            # Base hashed n-gram counts
            vec = np.zeros(hash_dim, dtype=np.float32)
            b = tok.encode("utf-8", errors="ignore")
            Lb = len(b)
            if Lb:
                for n in range(ngram_min, ngram_max + 1):
                    if Lb >= n:
                        for i in range(Lb - n + 1):
                            idx = _hash_ngram(b[i : i + n])
                            vec[idx] += 1.0

            # Lexical flags (8 dims)
            tlen = len(tok)
            if tlen == 0:
                alpha_ratio = digit_ratio = punct_ratio = 0.0
                num_caps = 0
            else:
                num_alpha = sum(ch.isalpha() for ch in tok)
                num_digit = sum(ch.isdigit() for ch in tok)
                num_punct = sum(ch in punct_set for ch in tok)
                alpha_ratio = num_alpha / tlen
                digit_ratio = num_digit / tlen
                punct_ratio = num_punct / tlen
                num_caps = sum(ch.isupper() for ch in tok)

            starts_boundary = 1.0 if tok.startswith(boundary_prefixes) else 0.0
            has_nonlatin = 0.0
            for ch in tok:
                if ch.isalpha():
                    try:
                        if "LATIN" not in unicodedata.name(ch, ""):
                            has_nonlatin = 1.0
                            break
                    except ValueError:
                        continue

            flags = np.array(
                [
                    starts_boundary,
                    alpha_ratio,
                    digit_ratio,
                    punct_ratio,
                    np.log1p(tlen),
                    np.log1p(len(b)),
                    float(num_caps > 0),
                    has_nonlatin,
                ],
                dtype=np.float32,
            )

            # Concatenate and L2-normalize
            out = np.concatenate([vec, flags], axis=0)
            norm = float(np.linalg.norm(out))
            if norm > 0:
                out /= norm
            return out

        # --- Build feature matrix [V, feat_dim] on CPU ---
        self.logger.info(f"Extracting hashed byte n-gram features (D={feat_dim})...")
        X = np.empty((V, feat_dim), dtype=np.float32)
        for i, tok in enumerate(id_to_tok):
            X[i] = token_features(tok)

        # --- Cluster on CPU ---
        self.logger.info(f"Clustering with MiniBatchKMeans (k={n_groups})...")
        try:
            km = _KMeans(
                n_clusters=int(n_groups),
                batch_size=4096,
                n_init="auto" if "auto" in str(getattr(_KMeans, "__init__", "")) else 10,
                random_state=rng_seed,
                max_iter=100,
                verbose=0,
            )
        except TypeError:
            # Compatibility for older sklearns without these kwargs
            km = _KMeans(n_clusters=int(n_groups), random_state=rng_seed)

        labels = km.fit_predict(X)

        # --- Map to id->group; mark specials as -1 ---
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        id2group: Dict[int, int] = {}
        for tid in range(V):
            id2group[tid] = -1 if tid in special_ids else int(labels[tid])

        self.logger.info("Done building tokenizer-only token groups.")
        return id2group