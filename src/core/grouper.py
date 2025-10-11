import os
import json
import torch
import numpy as np
from typing import Dict
from transformers import PreTrainedTokenizer, PreTrainedModel
from sklearn.cluster import MiniBatchKMeans


class Grouper:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, logger):
        """
        Group tokens based on their *semantic embeddings* rather than text form.

        Args:
            model: The pretrained model (e.g., GPT2, BERT).
            tokenizer: Corresponding tokenizer.
            logger: Logger object for progress messages.
        """
        logger.info("Start Grouping using model embeddings...")
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger

        # Extract the embedding layer once
        self.embeddings = self._get_embedding_matrix()

    # ------------------------------------------------------------
    # Utility: Save mapping id->group
    # ------------------------------------------------------------
    def write_id2group(self, data: Dict[int, int], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    # ------------------------------------------------------------
    # Main grouping wrapper
    # ------------------------------------------------------------
    @torch.no_grad()
    def group(self, cache_path: str = None, n_groups: int = 128) -> Dict[int, int]:
        """
        Cluster tokens by their model embeddings.
        """
        self.logger.info("Building token groups from embeddings...")

        id2group = self.build_token_groups(n_groups=n_groups)

        self.logger.info("Token grouping complete.")
        if cache_path:
            self.write_id2group(id2group, cache_path)

        return id2group

    # ------------------------------------------------------------
    # Core function: group using embeddings
    # ------------------------------------------------------------
    def build_token_groups(self, n_groups: int) -> Dict[int, int]:
        """
        Build token groups by clustering the model’s embedding vectors.
        Returns: dict {token_id: group_id}, specials → -1
        """
        V, D = self.embeddings.shape
        self.logger.info(f"Clustering {V} embeddings of dim {D} into {n_groups} groups...")

        # Normalize embeddings for stable clustering
        X = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        # Run MiniBatchKMeans on CPU
        km = MiniBatchKMeans(
            n_clusters=n_groups,
            batch_size=4096,
            n_init="auto",
            random_state=0,
            max_iter=100,
            verbose=0,
        )
        labels = km.fit_predict(X)

        # Mark special tokens (like <PAD>, <CLS>, <SEP>) with -1
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        id2group = {tid: (-1 if tid in special_ids else int(labels[tid])) for tid in range(V)}

        self.logger.info("Done building embedding-based token groups.")
        return id2group

    # ------------------------------------------------------------
    # Helper: Extract embeddings
    # ------------------------------------------------------------
    def _get_embedding_matrix(self) -> np.ndarray:
        """
        Return a [vocab_size, hidden_dim] matrix of embeddings from the model.
        Automatically moves model to CPU for memory efficiency.
        """
        self.model.eval()

        # Move to CPU for clustering (saves GPU memory)
        self.model.to("cpu")

        # Most HuggingFace models have embeddings under model.get_input_embeddings()
        emb_layer = self.model.get_input_embeddings()

        # Extract weight matrix and convert to numpy
        emb_matrix = emb_layer.weight.detach().cpu().numpy()

        self.logger.info(f"Loaded embedding matrix of shape {emb_matrix.shape}")
        return emb_matrix
