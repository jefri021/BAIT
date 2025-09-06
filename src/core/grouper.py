from datetime import datetime
import math
import gzip
import torch
import os
import json
from typing import Optional, List, Tuple, Dict, Sequence, Any, Iterable
from transformers import PreTrainedModel, PreTrainedTokenizer
from sklearn.cluster import KMeans
import torch.nn.functional as F
import hashlib



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

    
    def _simple_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        T: float
    ) -> torch.Tensor:
        """
        Get next-token probabilities in a single forward pass.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # logits shape: [batch_size, seq_len, vocab_size]
        logits = outputs.logits[:, -1, :]  

        # stable softmax over vocab
        output_probs = self.stable_softmax(logits, dim=-1, temperature=T)

        return output_probs
    

    def stable_softmax(self, logits, dim=-1, temperature=1.0):
        """Numerically stable softmax implementation"""
        # Subtract max for numerical stability
        logits = logits / temperature
        max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
        exp_logits = torch.exp(logits - max_logits)
        sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)
        
        # Add epsilon to prevent division by zero
        eps = 1e-12
        return exp_logits / (sum_exp + eps)
    


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
    @torch.no_grad()
    def group(
        self,
        cache_path: str = None,
        positions: Sequence[int] = (0, 1, 2),
        temps: Sequence[float] = (0.7, 1.0, 1.3),
        n_groups: int = 128,
        topk_mass: int = 5,
        chunk_tokens: int = 64,
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
        Memory-efficient reimplementation:
        - Streams over tokens and contexts.
        - Two-pass per-position z-score without storing the full [N, Dp].
        - Online combine across positions to avoid [P, N, Dp] stacking.
        """
        device = self.device
        panel_inputs = panel_inputs.to(device, non_blocking=True)
        panel_masks  = panel_masks.to(device, non_blocking=True)
        R, L = panel_inputs.shape

        # Resolve positions
        pos_list: List[int] = []
        for p in positions:
            pos = L - 1 if p == -1 else int(p)
            if not (0 <= pos < L):
                raise ValueError(f"Position {p} resolved to {pos} is out of range for L={L}.")
            pos_list.append(pos)

        # Token ids (exclude specials)
        specials = {
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        }
        allowed = [i for i in range(self.tokenizer.vocab_size) if i not in specials]
        token_ids = torch.tensor(allowed, dtype=torch.long, device=device, pin_memory=False)
        N = token_ids.numel()

        # Landmarks (optional) → ids
        landmark_ids: Optional[torch.Tensor] = None
        if landmarks:
            lm_ids: List[int] = []
            for lm in landmarks:
                tid = self.tokenizer.convert_tokens_to_ids(lm)
                if tid is None or tid < 0:
                    continue
                lm_ids.append(int(tid))
            if lm_ids:
                lm_ids = sorted(set(lm_ids))
                landmark_ids = torch.tensor(lm_ids, dtype=torch.long, device=device)

        # Small helper: online mean/var (Welford)
        class OnlineMV:
            def __init__(self, D: int, device):
                self.n = 0
                self.mean = torch.zeros(D, device=device)
                self.M2   = torch.zeros(D, device=device)
            def update(self, X: torch.Tensor):  # X: [*, D]
                # flatten first dimension
                x = X.reshape(-1, X.shape[-1])
                for row in x:
                    self.n += 1
                    delta = row - self.mean
                    self.mean += delta / self.n
                    self.M2   += delta * (row - self.mean)
            def finalize(self):
                var = self.M2 / max(self.n - 1, 1)
                std = torch.sqrt(var.clamp_min(1e-8))
                return self.mean, std

        # Memory knobs
        ctx_chunk = max(1, min(64, R))      # contexts per micro-batch (tune)
        mem_every = 8                       # empty_cache cadence (tune)

        # Mixed precision context
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext()

        @torch.inference_mode()
        def features_for_chunk_at_pos(z_chunk: torch.Tensor, pos: int) -> torch.Tensor:
            """
            Returns per-token, per-temp features for one position.
            Shape: [Nc, len(temps)*(4 + 2*K)]
            """
            Nc = z_chunk.numel()
            per_temp_blocks: List[torch.Tensor] = []

            # Stream over temps to keep peak small
            for t_idx, temp in enumerate(temps):
                # We will stream over contexts in micro-batches of size ctx_chunk
                # and aggregate H/top-k/landmark stats across R without building [Nc*R, L].
                H_rows_sum   = torch.zeros(Nc, device=device)
                H_rows_sum2  = torch.zeros(Nc, device=device)
                Tm_rows_sum  = torch.zeros(Nc, device=device)
                Tm_rows_sum2 = torch.zeros(Nc, device=device)

                if landmark_ids is not None and landmark_ids.numel() > 0:
                    K = landmark_ids.numel()
                    LP_rows_sum  = torch.zeros(Nc, K, device=device)
                    LP_rows_sum2 = torch.zeros(Nc, K, device=device)
                else:
                    K = 0
                    LP_rows_sum = LP_rows_sum2 = None

                # stream contexts
                for r0 in range(0, R, ctx_chunk):
                    r1 = min(r0 + ctx_chunk, R)
                    B  = r1 - r0

                    # Build a tiny batch [Nc*B, L] by repeating only the slice we need
                    base_inp = panel_inputs[r0:r1].repeat(Nc, 1)   # [Nc*B, L]
                    base_msk = panel_masks[r0:r1].repeat(Nc, 1)    # [Nc*B, L]
                    # substitute the target position with token ids
                    base_inp[:, pos] = z_chunk.repeat_interleave(B)
                    base_msk[:, pos] = 1

                    with amp_ctx:
                        probs = self._simple_generate(base_inp, base_msk, temp)  # [Nc*B, V]
                        p = probs.clamp_min(1e-12)

                        # per-row entropy and top-k mass
                        H  = (-p * p.log()).sum(dim=1)                               # [Nc*B]
                        Tm = torch.topk(probs, k=topk_mass, dim=1).values.sum(1)     # [Nc*B]

                        # reshape to [Nc, B] and accumulate mean/std components
                        Hb  = H.view(Nc, B);     Tmb = Tm.view(Nc, B)
                        H_rows_sum   += Hb.sum(1)
                        H_rows_sum2  += (Hb**2).sum(1)
                        Tm_rows_sum  += Tmb.sum(1)
                        Tm_rows_sum2 += (Tmb**2).sum(1)

                        if K:
                            LP = probs[:, landmark_ids]                              # [Nc*B, K]
                            LPb = LP.view(Nc, B, K)
                            LP_rows_sum  += LPb.sum(1)                               # [Nc, K]
                            LP_rows_sum2 += (LPb**2).sum(1)

                    # free ASAP
                    del base_inp, base_msk, probs
                    if (r0 // ctx_chunk) % mem_every == 0:
                        torch.cuda.empty_cache()

                # finalize across R contexts → mean/std
                denom = float(R)
                H_mean  = H_rows_sum  / denom
                H_std   = (H_rows_sum2/denom - H_mean**2).clamp_min(0).sqrt()
                Tm_mean = Tm_rows_sum / denom
                Tm_std  = (Tm_rows_sum2/denom - Tm_mean**2).clamp_min(0).sqrt()

                if K:
                    LP_mean = LP_rows_sum / denom                      # [Nc, K]
                    LP_var  = (LP_rows_sum2/denom - LP_mean**2).clamp_min(0)
                    LP_std  = LP_var.sqrt()
                    block = torch.cat(
                        [H_mean[:, None], H_std[:, None], Tm_mean[:, None], Tm_std[:, None], LP_mean, LP_std],
                        dim=1
                    )                                                  # [Nc, 4+2K]
                else:
                    block = torch.stack([H_mean, H_std, Tm_mean, Tm_std], dim=1)  # [Nc, 4]

                per_temp_blocks.append(block)

            return torch.cat(per_temp_blocks, dim=1)                   # [Nc, Dp]

        def random_project_stream(X_iter, Dp: int, out_dim: int):
            """
            X_iter yields [Nc, Dp] chunks already z-scored for the position.
            Apply Gaussian RP on the fly and L2-normalize; yield [Nc, d].
            """
            if out_dim is None or out_dim <= 0 or out_dim >= Dp:
                for X in X_iter:
                    Xf = F.normalize(X.to(torch.float32), p=2, dim=1)
                    yield Xf
                return
            W = torch.randn((Dp, out_dim), device=device, dtype=torch.float32) / math.sqrt(Dp)
            for X in X_iter:
                Y = X.to(torch.float32) @ W
                yield F.normalize(Y, p=2, dim=1)

        # -- Online combine across positions: keep running mean & std for [mean || std] --
        # We don't know Dp yet; compute it from the first small probe
        with torch.inference_mode():
            probe = features_for_chunk_at_pos(token_ids[:min(8, N)], pos_list[0])
            Dp = probe.shape[1]
            del probe
        d_after = compress_dim if (compress_dim and 0 < compress_dim < 2*Dp) else 2*Dp

        # Running stats over positions for *final* feature X_all (after z-score and RP)
        mv_final = OnlineMV(D=d_after, device=device)

        # Process each position with TWO passes to z-score without storing [N, Dp]
        for p_idx, pos in enumerate(pos_list):
            # PASS 1: column stats across all tokens (streamed by chunks)
            mv_pos = OnlineMV(D=Dp, device=device)
            for start in range(0, N, chunk_tokens):
                end = min(start + chunk_tokens, N)
                z_chunk = token_ids[start:end]
                Xp_chunk = features_for_chunk_at_pos(z_chunk, pos)  # [Nc, Dp]
                mv_pos.update(Xp_chunk)
                del Xp_chunk
            mu_p, sd_p = mv_pos.finalize()

            # PASS 2: z-score, then build [mean||std] across positions ONLINE with Welford
            def zscored_chunks():
                for start in range(0, N, chunk_tokens):
                    end = min(start + chunk_tokens, N)
                    z_chunk = token_ids[start:end]
                    Xp = features_for_chunk_at_pos(z_chunk, pos)                # [Nc, Dp]
                    Xp = (Xp - mu_p) / sd_p                                     # z-score per position
                    yield Xp
                    del Xp

            # Combine across positions: maintain running mean & std over positions for each token
            # We compute mean & std across positions by accumulating per-chunk over tokens:
            # For each chunk, we need mean and std across positions; do Welford per-token vector.
            # Implementation trick: keep per-token accumulators in CPU numpy at the very end is costly,
            # so we fold positions directly into mv_final after projecting [mean||std] for this position only.
            # To do that, we first compute RP(Xp_z) for position p, then maintain *two* running stats
            # across positions (mean and squared terms) per token offline; but to stay memory-light,
            # we approximate [mean||std] across positions by accumulating mean and M2 across positions on-the-fly.

            # We'll first accumulate mean and M2 for this position's *post-RP* vectors per token,
            # then after finishing all positions, mv_final will actually be the combination across tokens,
            # not across positions. To keep the original spec ([mean||std] across positions), we do it explicitly here.

            # Per-token accumulators across positions require storing [N, d] * 2 (mean,M2) -> heavy.
            # Alternative: compute [mean||std] across positions AFTER all positions by a second loop (too slow).
            # Pragmatic compromise: compute [mean||std] across positions in-place *per chunk*, keeping only
            # chunk-local accumulators, then immediately feed the concatenated [mean||std] to mv_final.
            # This matches the original behavior without keeping full [N, d].

            # Initialize chunk-local accumulators
            first_pos = (p_idx == 0)
            if first_pos:
                # Create temporary per-token buffers on disk? Not necessary:
                # We'll keep running per-chunk stats across positions in a dict keyed by start index.
                pos_acc = {}
            # For this position, just iterate zscored chunks and RP -> emit to a staging dict
            rp_iter = random_project_stream(zscored_chunks(), Dp, compress_dim)

            # Store projected chunks in a list to then compute [mean||std] across positions.
            # To avoid holding multiple positions at once, we materialize only the current position (one at a time),
            # and merge into chunk-level running stats stored on CPU to minimize GPU peaks.
            chunk_idx = 0
            for start in range(0, N, chunk_tokens):
                Y = next(rp_iter)  # [Nc, d]
                Yc = Y.detach().to('cpu', copy=True)  # move off GPU quickly
                if first_pos:
                    # initialize accumulators for this chunk
                    pos_acc[start] = {
                        "n": 0,
                        "mean": torch.zeros_like(Yc),
                        "M2": torch.zeros_like(Yc),
                    }
                acc = pos_acc[start]
                acc["n"] += 1
                delta = Yc - acc["mean"]
                acc["mean"] += delta / acc["n"]
                acc["M2"]   += delta * (Yc - acc["mean"])
                del Y, Yc
                chunk_idx += 1
                if (chunk_idx % mem_every) == 0:
                    torch.cuda.empty_cache()

            # If this was the last position, finalize [mean||std] for each chunk and feed to mv_final
            if p_idx == len(pos_list) - 1:
                for start in range(0, N, chunk_tokens):
                    acc = pos_acc[start]
                    mean_p = acc["mean"]                       # [Nc, d] on CPU
                    std_p  = torch.sqrt((acc["M2"] / max(acc["n"] - 1, 1)).clamp_min(1e-8))
                    X_all_chunk = torch.cat([mean_p, std_p], dim=1).to(device)  # [Nc, 2d]
                    mv_final.update(X_all_chunk)              # online stats over tokens
                    del X_all_chunk
                del pos_acc
            # proceed to next position

        # Final feature stats across tokens (not strictly needed for KMeans, but we keep normalized vectors)
        mu_final, sd_final = mv_final.finalize()

        # Stream a last time to produce the final matrix for clustering, but keep it chunked to limit peak.
        # We’ll write chunks into a CPU list before numpy handoff.
        Xc_cpu_chunks: List[torch.Tensor] = []

        # We must reproduce the same pipeline to emit final [N, 2d]; use the pos_acc path again
        # to avoid storing all positions simultaneously.
        for start in range(0, N, chunk_tokens):
            # rebuild per-chunk accumulators across positions
            npos = 0
            mean_c = None
            M2_c   = None

            for pos in pos_list:
                # z-score chunks for this position and project; but only the specific chunk [start:end]
                end = min(start + chunk_tokens, N)
                z_chunk = token_ids[start:end]

                def zscored_single():
                    Xp = features_for_chunk_at_pos(z_chunk, pos)  # [Nc, Dp]
                    # Recompute mu_p/sd_p for this position quickly:
                    # For efficiency, cache per-position stats in a dict the first time we computed them.
                    # To keep code simple and still memory-safe, recompute with a quick tiny pass:
                    # (If you want max speed, cache mu/sd in a dict during the earlier pass.)
                    # --- BEGIN fast recompute of mu/sd for this position over this chunk only ---
                    # NOTE: True per-position zscore used global mu/sd across all tokens.
                    # For exact reproducibility, you can store mu/sd from earlier; omitted here for memory.
                    # As a practical compromise, we use the earlier mu_p/sd_p computed for the *whole* position,
                    # but we didn't retain them. If you need exactness, add a small dict to store them.
                    # For now, we recompute global mu/sd per position once (cheap) and cache:
                    return Xp  # we’ll normalize using cached stats below

                # Cache global mu/sd per position from earlier pass:
                # For correctness and speed, let’s compute once per position and reuse.
                # We actually *did* compute mu_p, sd_p above, but didn’t keep them.
                # To keep memory small yet be correct, let’s compute and keep them in a dict:
                if "_pos_stats" not in locals():
                    _pos_stats = {}
                if pos not in _pos_stats:
                    # recompute once (streamed): get global mu/sd for this position
                    mv_pos_f = OnlineMV(D=Dp, device=device)
                    for s2 in range(0, N, chunk_tokens):
                        e2 = min(s2 + chunk_tokens, N)
                        Xp2 = features_for_chunk_at_pos(token_ids[s2:e2], pos)
                        mv_pos_f.update(Xp2)
                        del Xp2
                    _pos_stats[pos] = mv_pos_f.finalize()

                mu_p, sd_p = _pos_stats[pos]
                Xp = features_for_chunk_at_pos(z_chunk, pos)
                Xp = (Xp - mu_p) / sd_p
                for Y in random_project_stream([Xp], Dp, compress_dim):
                    Yc = Y.detach().to('cpu', copy=True)
                del Xp, Y, Yc  # Yc was deleted; fix: keep it
                # (fix deletion order)
                Yc = random_project_stream([ (features_for_chunk_at_pos(z_chunk, pos) - mu_p)/sd_p ], Dp, compress_dim)
                Yc = next(Yc).detach().to('cpu', copy=True)

                npos += 1
                if mean_c is None:
                    mean_c = torch.zeros_like(Yc)
                    M2_c   = torch.zeros_like(Yc)
                delta = Yc - mean_c
                mean_c += delta / npos
                M2_c   += delta * (Yc - mean_c)
                del Yc
                if npos % mem_every == 0:
                    torch.cuda.empty_cache()

            std_c = torch.sqrt((M2_c / max(npos - 1, 1)).clamp_min(1e-8))
            X_all_chunk = torch.cat([mean_c, std_c], dim=1)  # on CPU
            Xc_cpu_chunks.append(X_all_chunk)
            del mean_c, M2_c, X_all_chunk, std_c

        # Concatenate CPU chunks and handoff to sklearn
        Xc_cpu = torch.cat(Xc_cpu_chunks, dim=0)  # [N, 2d] on CPU
        Xc_np = Xc_cpu.numpy()
        del Xc_cpu, Xc_cpu_chunks
        torch.cuda.empty_cache()

        self.logger.info("Clustering token groups...")
        km = KMeans(n_clusters=n_groups, n_init="auto")
        labels_np = km.fit_predict(Xc_np)

        id2group = {int(tid.item()): int(lbl) for tid, lbl in zip(token_ids, torch.from_numpy(labels_np))}
        # Specials -> -1
        for vid in range(self.tokenizer.vocab_size):
            if vid not in id2group:
                id2group[vid] = -1
        return id2group