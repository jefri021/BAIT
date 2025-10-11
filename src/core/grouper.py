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
    # Utility: Save mapping id->group and cluster representatives
    # ------------------------------------------------------------
    def write_group_results(self, id2group: Dict[int, int], cluster_reps: list, path: str) -> None:
        """
        Save both the token->group mapping and the representative tokens per cluster as JSON.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_data = {
            "id2group": id2group,
            "cluster_representatives": [
                {
                    "cluster_id": rep["cluster_id"],
                    "token_ids": [int(i) for i in rep["token_ids"]]
                }
                for rep in cluster_reps
            ],
        }
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)
        self.logger.info(f"Saved grouping results to {path}")

    # ------------------------------------------------------------
    # Main grouping wrapper
    # ------------------------------------------------------------
    @torch.no_grad()
    def group(self, cache_dir: str = None, n_groups: int = 128):
        """
        Cluster tokens by their model embeddings and return both:
        - id2group: mapping token_id -> group_id
        - cluster_reps: representative tokens for each cluster
        """
        self.logger.info("Building token groups from embeddings...")

        id2group, cluster_reps = self.build_token_groups(n_groups=n_groups)

        self.logger.info("Token grouping complete.")
        if cache_dir:
            self.write_group_results(id2group, cluster_reps, cache_dir)

        return id2group, cluster_reps


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

        # Normalize embeddings for stable clustering (handle zero and near-zero vectors safely)
        eps = 1e-12  # stability constant

        # Compute row norms
        norms = np.linalg.norm(self.embeddings, axis=1)

        # Replace zeros and NaNs with 1.0 before division
        safe_norms = np.copy(norms)
        safe_norms[~np.isfinite(safe_norms)] = 1.0
        safe_norms[safe_norms < eps] = 1.0

        # Perform normalization safely
        X = np.divide(self.embeddings, safe_norms[:, None], out=np.zeros_like(self.embeddings), where=safe_norms[:, None] != 0)

        # Clean up any residual NaN/Inf (just in case)
        X = np.nan_to_num(X, copy=False)

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
        centers = km.cluster_centers_

        cluster_reps = self.get_cluster_representatives(X, labels, centers)

        # Mark special tokens (like <PAD>, <CLS>, <SEP>) with -1
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        id2group = {tid: (-1 if tid in special_ids else int(labels[tid])) for tid in range(V)}

        self.logger.info("Done building embedding-based token groups.")
        return id2group, cluster_reps

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
        # self.model.to("cpu")

        # Most HuggingFace models have embeddings under model.get_input_embeddings()
        emb_layer = self.model.get_input_embeddings()

        # Extract weight matrix and convert to numpy
        emb_matrix = emb_layer.weight.detach().cpu().numpy()

        self.logger.info(f"Loaded embedding matrix of shape {emb_matrix.shape}")
        return emb_matrix
    

    def get_cluster_representatives(self, embeddings, labels, centers, top_k=1):
        cluster_reps = []
        for i in range(len(centers)):
            cluster_indices = np.where(labels == i)[0]
            cluster_embs = embeddings[cluster_indices]

            # Distance from center to members
            dists = np.linalg.norm(cluster_embs - centers[i], axis=1)
            nearest_indices = cluster_indices[np.argsort(dists)[:top_k]]

            cluster_reps.append({
                "cluster_id": i,
                "token_ids": nearest_indices
            })
        return cluster_reps
