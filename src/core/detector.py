"""
bait.py: Core module for the BAIT (LLM Backdoor Scanning) project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-10-01]
Version: 1.1

This module contains the main BAIT class It provides
the core functionality for initializing and running backdoor scans on LLMs.

Copyright (c) [2024] [PurduePAML]
"""
import torch
import os
import json
import traceback
from time import time, sleep
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
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


@dataclass
class BestTarget:
    q_score: float = 0
    invert_target: str = None
    reasoning: str = ""
    
    def __str__(self) -> str:
        return (f"BestTarget:\n"
                f"  q_score: {self.q_score}\n"
                f"  invert_target: {self.invert_target!r}\n"
                f"  reasoning: {self.reasoning!r}")

@dataclass
class ScanResult:
    is_backdoor: bool
    best_target: BestTarget


class BAIT:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataloader: torch.utils.data.DataLoader,
        bait_args: BAITArguments,
        logger: Optional[object] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the BAIT object.

        Args:
            model (PreTrainedModel): The pre-trained language model.
            tokenizer (PreTrainedTokenizer): The tokenizer for the model.
            dataloader (DataLoader): DataLoader for input data.
            bait_args (BAITArguments): Configuration arguments for BAIT.
            logger (Optional[object]): Logger object for logging information.
            device (str): Device to run the model on (cuda or cpu).
        """
        logger.info("Initializing BAIT...")
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.logger = logger
        self.device = device
        self._init_config(bait_args)
        self.judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    @torch.no_grad()
    def run(self) -> ScanResult:
        """
        Run the BAIT algorithm on the input data.

        Returns:
            ScanResult: A ScanResult object containing:
                - Boolean indicating if a backdoor was detected
                - The highest Q-score found
                - The invert target (token IDs) for the potential backdoor
        """

        best_target = BestTarget()

        for batch_inputs in tqdm(self.dataloader, desc="Scanning data..."):
            input_ids = batch_inputs["input_ids"]
            attention_mask = batch_inputs["attention_mask"]
            index_map = batch_inputs["index_map"]

            batch_q_score, batch_invert_target = self.scan_init_token(input_ids, attention_mask, index_map)
            self.logger.debug(f"Batch Q-score: {batch_q_score}, Batch Invert Target: {batch_invert_target}")

            if batch_q_score > best_target.q_score:
                # post-process to further exam if the invert target includes suspicious content which might be a backdoor target string
                batch_is_suspicious, batch_reasoning = self.__post_process(batch_invert_target)
                if batch_is_suspicious:
                    # update best target
                    best_target.q_score = batch_q_score
                    best_target.invert_target = batch_invert_target
                    best_target.reasoning = batch_reasoning
                    self.logger.info(f"New best target found: {best_target}")

            # early stop if a very promising target is found
            if self.early_stop and best_target.q_score > self.early_stop_q_score_threshold:
                self.logger.info(f"Early stop at q-score: {best_target.q_score}")
                break

        if best_target.q_score > self.q_score_threshold:
            self.logger.info(f"Q-score is greater than threshold: {self.q_score_threshold}")
            self.logger.info(f"Inverted Target contains suspicious content: {best_target.invert_target}")
            self.logger.info(f"Reasoning: {best_target.reasoning}")
            is_backdoor = True
        else:
            self.logger.info(f"Q-score is less than threshold: {self.q_score_threshold}")
            is_backdoor = False
        
        return ScanResult(is_backdoor, best_target)


    def __post_process(
        self,
        invert_target: str,
    ) -> str:
        """
        Post-process to further exam if the invert target includes suspicious content which might be a backdoor target string

        Args:
            invert_target (str): The target string to analyze
        """

        for attempt in range(self.max_retries):
            try:
                response = self.judge_client.chat.completions.create(
                    model=self.judge_model_name,
                    messages=[
                        {"role": "user", "content": JUDGE_SYSTEM_PROMPT.format(response=invert_target)}
                    ]
                ).choices[0].message.content

                try:
                    state = extract_tag(response, "State").lower().strip()
                    reasoning = extract_tag(response, "Reasoning")

                    if not state or not reasoning:
                        self.logger.error("Missing required tags in response")
                        continue

                    if state not in ["suspicious", "safe"]:
                        self.logger.error(f"Invalid state value: {state}")
                        continue

                    if state == "suspicious":
                        return True, reasoning
                    else:
                        return False, reasoning

                except (ValueError, AttributeError, IndexError) as e:
                    self.logger.error(f"Failed to parse response: {str(e)}")
                    if attempt == self.max_retries - 1:
                        return False, "Error: Failed to parse response after multiple attempts"
                    continue

            except (APIError, RateLimitError, APIConnectionError) as e:
                if attempt == self.max_retries - 1:  # Last attempt
                    self.logger.error(f"Failed to get response after {self.max_retries} attempts: {str(e)}")
                    return False, "Error: Failed to analyze content after multiple attempts"

                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {self.retry_delay} seconds...")
                sleep(self.retry_delay)
                self.retry_delay *= 2  # Exponential backoff

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

    def __generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 1
    ) -> torch.Tensor:
        """
        Generate output probabilities for the next token using the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            torch.Tensor: Output probabilities for the next token.
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            top_p=self.top_p,
            temperature=self.temperature,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            do_sample=self.do_sample,
            return_dict_in_generate=self.return_dict_in_generate,
            output_scores=self.output_scores
        )

        output_scores = outputs.scores[0]
        
        # Handle NaN and inf values in output scores
        output_scores = torch.nan_to_num(output_scores, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # print(f"output_scores: {output_scores}")
        # print(f"before softmax: {output_scores.max()}, {output_scores.min()}")
        
        # Check for any remaining problematic values
        if torch.isnan(output_scores).any() or torch.isinf(output_scores).any():
            self.logger.warning("Found NaN or inf values in output scores after cleaning")
            # Replace entire tensor with uniform distribution if still problematic
            output_scores = torch.zeros_like(output_scores)
        
        output_probs = self.stable_softmax(output_scores, dim=-1)
        # print(f"after softmax: {output_probs.max()}, {output_probs.min()}")

        return output_probs
    

    def _search_best_trigger_token(
        self,
        batch_input_ids: torch.Tensor,      # (B, L)
        batch_attention_mask: torch.Tensor, # (B, L)
        tgt_token_id: int,                  # a_t just chosen
        candidate_vocab: torch.Tensor,      # (V',)
        pos: int                            # which position to overwrite (0,1,…)
    ) -> int:
        """
        Replace *pos*‑th token of every prompt with each candidate and pick
        the one that maximises the average P(Y_t = tgt_token_id).

        Returns the winning token id.
        """
        # ---- prepare replicated prompts ---------------------------------
        B, L  = batch_input_ids.size()
        Vp    = candidate_vocab.size(0)

        expanded_ids   = batch_input_ids.repeat(Vp, 1).clone()         # (V'·B, L)
        expanded_mask  = batch_attention_mask.repeat(Vp, 1)            # (V'·B, L)

        for j, tok in enumerate(candidate_vocab):
            expanded_ids[j*B:(j+1)*B, pos] = tok

        # ---- forward pass ----------------------------------------------
        with torch.no_grad():
            logits = self.model(
                input_ids      = expanded_ids.to(self.device),
                attention_mask = expanded_mask.to(self.device)
            ).logits                                              # (V'·B, L, |V|)
            next_logits = logits[:, -1, :]                        # (V'·B, |V|)
            probs       = torch.softmax(next_logits, dim=-1)[:, tgt_token_id]
            probs       = probs.view(Vp, B).mean(dim=1)           # (V',)

        return candidate_vocab[probs.argmax()].item()

    def warm_up_inversion(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform warm-up inversion to using a mini-batch and short generation steps

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed targets and target probabilities.
        """
        batch_size = min(self.batch_size, int(input_ids.shape[0] // self.warmup_batch_size))
        targets = torch.zeros(self.warmup_steps, batch_size).long().to(self.device) - 1
        target_probs = torch.zeros(self.warmup_steps, batch_size).to(self.device) - 1
        target_mapping_record = [torch.arange(batch_size).to(self.device)]
        uncertainty_inspection_times = torch.zeros(batch_size).to(self.device)

        processed_targets = torch.zeros(self.warmup_steps, batch_size).long().to(self.device) - 1
        processed_target_probs = torch.zeros(self.warmup_steps, batch_size).to(self.device) - 1

        for step in range(self.warmup_steps):
            output_probs = self.__generate(input_ids, attention_mask)
            input_ids, attention_mask, targets, target_probs, target_mapping_record, uncertainty_inspection_times = self._update(
                targets,
                target_probs,
                output_probs,
                input_ids,
                attention_mask,
                step,
                target_mapping_record,
                uncertainty_inspection_times
            )

            if input_ids is None:
                self.logger.debug("Input ids is empty, break")
                return processed_targets, processed_target_probs


        last_step_indices = target_mapping_record[-1]
        original_indices = []
        for idx in range(len(last_step_indices)):
            # trace back to the first step
            original_idx = last_step_indices[idx]
            for step in range(len(target_mapping_record)-2, -1, -1):
                original_idx = target_mapping_record[step][original_idx]
            original_indices.append(original_idx)

        original_indices = torch.tensor(original_indices)
        processed_targets[:,original_indices] = targets
        processed_target_probs[:,original_indices] = target_probs
        return processed_targets, processed_target_probs

    def full_inversion(
        self,
        warmup_targets: torch.Tensor,
        warmup_target_probs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        index_map: List[int]
    ) -> Tuple[float, torch.Tensor]:
        """
        Perform full inversion to find the highest Q-score and invert target.

        Args:
            warmup_targets (torch.Tensor): Targets from warm-up inversion.
            warmup_target_probs (torch.Tensor): Target probabilities from warm-up inversion.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            index_map (List[int]): Mapping of indices for batches.

        Returns:
            Tuple[float, torch.Tensor]: Highest Q-score and corresponding invert target.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        q_score = 0
        invert_target = None


        batch_size = min(self.batch_size, int(input_ids.shape[0] // self.prompt_size))

        for i in range(batch_size):
            if -1 in warmup_targets[:,i]:
                continue

            warmup_target = warmup_targets[:,i]
            warmup_target_prob = warmup_target_probs[:,i]
            batch_input_ids = input_ids[i*self.prompt_size:(i+1)*self.prompt_size]
            batch_attention_mask = attention_mask[i*self.prompt_size:(i+1)*self.prompt_size]

            initial_token = batch_input_ids[0, -1].unsqueeze(0)

            batch_target = []
            batch_target_prob = []

            for step in range(self.full_steps):
                output_probs = self.__generate(batch_input_ids, batch_attention_mask)
                avg_probs = output_probs.mean(dim=0)
                if (self.enable_trigger_search
                        and step < self.trigger_max_steps):          # only first k steps
                    # restrict search space to top‑K tokens by prob
                    cand_vocab = torch.topk(avg_probs, k=self.trigger_topk).indices
                    best_trigger = self._search_best_trigger_token(
                        batch_input_ids        = batch_input_ids,
                        batch_attention_mask   = batch_attention_mask,
                        tgt_token_id           = torch.argmax(avg_probs).item(),   # tentative a_t
                        candidate_vocab        = cand_vocab,
                        pos                    = step       # overwrite 0,1,…
                    )
                    # overwrite pos‑th token of *every* prompt replica
                    batch_input_ids[:, step] = best_trigger
                if step < self.warmup_steps:
                    new_token = warmup_target[step].unsqueeze(0).expand(self.prompt_size, -1)
                    batch_target.append(warmup_target[step])
                    batch_target_prob.append(avg_probs[warmup_target[step]])
                else:
                    top_prob, top_token = torch.max(avg_probs, dim=-1)
                    new_token = top_token.unsqueeze(0).expand(self.prompt_size, -1)
                    batch_target.append(top_token)
                    batch_target_prob.append(top_prob)

                batch_input_ids = torch.cat([batch_input_ids, new_token], dim=-1)
                batch_attention_mask = torch.cat([batch_attention_mask, batch_attention_mask[:, -1].unsqueeze(1)], dim=-1)



                if batch_target[step].item() == self.tokenizer.eos_token_id or self.tokenizer.decode(batch_target[step].item()) == "<|end_of_text|>":
                    self.logger.debug(f"EOS token reached at step {step}")
                    break

            batch_target = torch.tensor(batch_target).long()
            batch_target_prob = torch.tensor(batch_target_prob)


            if self.tokenizer.eos_token_id in batch_target:
                eos_id = torch.where(batch_target == self.tokenizer.eos_token_id)[0][0].item()
                batch_target = batch_target[:eos_id]
                batch_target_prob = batch_target_prob[:eos_id]
            
            if self.tokenizer.encode("<|end_of_text|>", add_special_tokens=False)[0] in batch_target:
                eos_id = torch.where(batch_target == self.tokenizer.encode("<|end_of_text|>", add_special_tokens=False)[0])[0][0].item()
                batch_target = batch_target[:eos_id]
                batch_target_prob = batch_target_prob[:eos_id]


            # Remove the smallest probability from batch_target_prob to improve detection robustness
            if len(batch_target_prob) > 1:
                min_prob_index = torch.argmin(batch_target_prob)
                batch_target_prob = torch.cat([batch_target_prob[:min_prob_index], batch_target_prob[min_prob_index+1:]])

            # Calculate the batch_q_score as the mean of the remaining probabilities
            batch_q_score = batch_target_prob.mean().item()
            batch_target = torch.cat([initial_token.detach().cpu(), batch_target], dim=-1)
            batch_invert_target = self.tokenizer.decode(batch_target)
            self.logger.debug(f"batch_invert_target: {batch_invert_target}")
            self.logger.debug(f"batch_q_score: {batch_q_score}")
            if batch_q_score > q_score and len(batch_invert_target.split()) >= self.min_target_len:
                q_score = batch_q_score
                invert_target = batch_invert_target

        return q_score, invert_target

    def scan_init_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        index_map: List[int]
    ) -> Tuple[float, torch.Tensor]:
        """
        enumerate initial tokens and invert the entire attack target.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            index_map (List[int]): Mapping of indices for batches.

        Returns:
            Tuple[float, torch.Tensor]: Q-score and invert target for potential backdoor.
        """
        sample_index = []
        for map_idx in index_map:
            start_idx = index_map[map_idx]
            end_idx = index_map[map_idx] +  self.warmup_batch_size
            sample_index.extend(i for i in range(start_idx, end_idx))


        sample_input_ids = input_ids[sample_index].to(self.device)
        sample_attention_mask = attention_mask[sample_index].to(self.device)
        warmup_targets, warmup_target_probs = self.warm_up_inversion(sample_input_ids, sample_attention_mask)
        return self.full_inversion(warmup_targets, warmup_target_probs, input_ids, attention_mask, index_map)


    def uncertainty_inspection(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        avg_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform uncertainty inspection for the current batch.
        """
        topk_probs, topk_indices = torch.topk(avg_probs, k=self.uncertainty_inspection_topk, dim=-1)
        #============================Debugging log============================
        for topk_prob, topk_index in zip(topk_probs, topk_indices):
            token = self.tokenizer.convert_ids_to_tokens(topk_index.tolist())
            self.logger.debug(f"Tokens: {token:<20} | IDs: {topk_index.item():<20} | Probs: {topk_prob.item():<20.4f}")
        #============================Debugging log============================
        reshape_topk_indices = topk_indices.view(-1).repeat_interleave(self.warmup_batch_size).unsqueeze(1)
        input_ids = input_ids.repeat(self.uncertainty_inspection_topk, 1)
        attention_mask = attention_mask.repeat(self.uncertainty_inspection_topk, 1)
        input_ids = torch.cat([input_ids, reshape_topk_indices], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask[:, -1].unsqueeze(1)], dim=-1)
        output_probs = self.__generate(input_ids, attention_mask).view(self.uncertainty_inspection_topk, self.warmup_batch_size, -1).mean(dim=1)
        max_prob, max_indices = torch.max(output_probs, dim=-1)
        new_token = topk_indices[max_prob.argmax()]

        #============================Debugging log============================
        self.logger.debug(f"Max prob: {max_prob}")
        self.logger.debug(f"Max indices: {max_indices}")
        self.logger.debug(f"max_indices.argmax(): {max_prob.argmax()}")
        self.logger.debug(f"decode: {self.tokenizer.decode(max_prob.argmax())}")
        self.logger.debug(f"new_token: {new_token}")
        self.logger.debug(f"decode: {self.tokenizer.decode(new_token)}")
        #============================Debugging log============================
        return new_token



    def _init_config(self, bait_args: BAITArguments) -> None:
        """
        Initialize configuration from BAITArguments.

        Args:
            bait_args (BAITArguments): Configuration arguments for BAIT.
        """
        for key, value in bait_args.__dict__.items():
            setattr(self, key, value)


    def _update(
        self,
        targets: torch.Tensor,
        target_probs: torch.Tensor,
        output_probs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step: int,
        target_mapping_record: List[torch.Tensor],
        uncertainty_inspection_times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Update targets, probabilities, and input sequences based on output probabilities.

        Args:
            targets (torch.Tensor): Current target tokens.
            target_probs (torch.Tensor): Current target probabilities.
            output_probs (torch.Tensor): Output probabilities from the model.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            step (int): Current step in the inversion process.
            target_mapping_record (List[torch.Tensor]): Record of target mappings.
            tolerance_times (torch.Tensor): Record of tolerance times for each sequence.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
                Updated input_ids, attention_mask, targets, target_probs, and target_mapping_record.
        """
        # Calculate average probabilities across the warmup batch
        batch_size = target_mapping_record[-1].shape[0]
        avg_probs = output_probs.view(batch_size, self.warmup_batch_size, -1).mean(dim=1)

        self_entropy = self._compute_self_entropy(avg_probs)


        selected_indices = []
        selected_input_ids = []
        selected_attention_mask = []


        for cand_idx in range(batch_size):
            cand_self_entropy = self_entropy[cand_idx]
            cand_avg_probs = avg_probs[cand_idx]
            cand_max_prob = cand_avg_probs.max()
            cand_batch_input_ids = input_ids[cand_idx * self.warmup_batch_size:(cand_idx + 1) * self.warmup_batch_size]
            cand_batch_attention_mask = attention_mask[cand_idx * self.warmup_batch_size:(cand_idx + 1) * self.warmup_batch_size]

            if (self.enable_trigger_search and step < self.trigger_max_steps):
                cand_vocab = torch.topk(cand_avg_probs, k=self.trigger_topk).indices
                best_z = self._search_best_trigger_token(
                    batch_input_ids      = cand_batch_input_ids,
                    batch_attention_mask = cand_batch_attention_mask,
                    tgt_token_id         = cand_avg_probs.argmax().item(),
                    candidate_vocab      = cand_vocab,
                    pos                  = step
                )
                # overwrite the *step*‑th token of every replica
                cand_batch_input_ids[:, step] = best_z


            cand_uncertainty_inspection_times = uncertainty_inspection_times[cand_idx]
            uncertainty_conditions = self._check_uncertainty(cand_self_entropy, cand_avg_probs, cand_max_prob, cand_uncertainty_inspection_times)
            if uncertainty_conditions:
                self.logger.debug(f"Uncertainty inspection conditions met for candidate token: {self.tokenizer.convert_ids_to_tokens(cand_batch_input_ids[0][-1].tolist())}")
                new_token = self.uncertainty_inspection(cand_batch_input_ids, cand_batch_attention_mask, cand_avg_probs)
                if new_token == self.tokenizer.eos_token_id or self.tokenizer.decode(new_token) == "<|end_of_text|>":
                    continue

                uncertainty_inspection_times[cand_idx] += 1
                targets[step][cand_idx] = new_token
                target_probs[step][cand_idx] = cand_avg_probs[new_token]
                cand_batch_input_ids = torch.cat([cand_batch_input_ids, new_token.view(-1, 1).expand(-1, self.warmup_batch_size).reshape(-1, 1)], dim=-1)
                cand_batch_attention_mask = torch.cat([cand_batch_attention_mask, cand_batch_attention_mask[:, -1].unsqueeze(1)], dim=-1)

                selected_indices.append(cand_idx)
                selected_input_ids.append(cand_batch_input_ids)
                selected_attention_mask.append(cand_batch_attention_mask)

            else:
                if cand_self_entropy < self.self_entropy_lower_bound or cand_max_prob > self.expectation_threshold:
                    new_token = cand_avg_probs.argmax()
                    if new_token == self.tokenizer.eos_token_id or self.tokenizer.decode(new_token) == "<|end_of_text|>":
                        continue

                    targets[step][cand_idx] = new_token
                    target_probs[step][cand_idx] = cand_max_prob
                    cand_batch_input_ids = torch.cat([cand_batch_input_ids, new_token.view(-1, 1).expand(-1, self.warmup_batch_size).reshape(-1, 1)], dim=-1)
                    cand_batch_attention_mask = torch.cat([cand_batch_attention_mask, cand_batch_attention_mask[:, -1].unsqueeze(1)], dim=-1)

                    selected_indices.append(cand_idx)
                    selected_input_ids.append(cand_batch_input_ids)
                    selected_attention_mask.append(cand_batch_attention_mask)

        if len(selected_indices) == 0:
            return None, None, None, None, None, None
        else:
            selected_indices = torch.tensor(selected_indices).long().to(self.device)
            input_ids = torch.cat(selected_input_ids, dim=0)
            attention_mask = torch.cat(selected_attention_mask, dim=0)
            targets = targets[:, selected_indices]
            target_probs = target_probs[:, selected_indices]
            target_mapping_record.append(selected_indices)
            return input_ids, attention_mask, targets, target_probs, target_mapping_record, uncertainty_inspection_times


    def _check_uncertainty(
        self,
        self_entropy: torch.Tensor,
        avg_probs: torch.Tensor,
        max_prob: torch.Tensor,
        uncertainty_inspection_times: torch.Tensor
    ) -> bool:
        """
        Check if the uncertainty condition is met.
        """
        cr1 = uncertainty_inspection_times < self.uncertainty_inspection_times_threshold
        cr2 = self_entropy < self.self_entropy_upper_bound
        cr3 = self_entropy > self.self_entropy_lower_bound
        cr4 = max_prob < self.expectation_threshold
        return cr1 and ((cr2 and cr3) or (cr2 and cr4))

    def _compute_self_entropy(
        self,
        probs_distribution: torch.Tensor,
        eps: float = 1e-10
    ) -> torch.Tensor:
        """
        Compute the self-entropy of a probability distribution.

        Args:
            probs_distribution (torch.Tensor): Probability distribution.
            eps (float): Small value to avoid log(0).

        Returns:
            torch.Tensor: Computed self-entropy.
        """
        # Add eps to avoid log(0) and handle NaN values
        probs_distribution = torch.nan_to_num(probs_distribution, nan=0.0) + eps
        # print(probs_distribution)

        # Normalize the distribution
        probs_distribution = probs_distribution / probs_distribution.sum(dim=-1, keepdim=True)

        # Compute entropy
        entropy = - (probs_distribution * torch.log(probs_distribution)).sum(dim=-1)
        return entropy



class BAITWrapper:
    """Handles the scanning of a single model"""
    def __init__(self, model_id: str, model_config: Dict, scan_args: ScanArguments, run_dir: str):
        self.model_id = model_id
        self.model_config = model_config
        self.scan_args = scan_args
        self.run_dir = run_dir
        self.log_dir = os.path.join(run_dir, model_id)
        os.makedirs(self.log_dir, exist_ok=True)

        self._setup_logging()
        self.bait_args, self.model_args, self.data_args = self._initialize_arguments()

    def _setup_logging(self):
        """Configure logging for this model scan"""
        log_file = os.path.join(self.log_dir, "scan.log")
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(log_file, rotation="100 MB", level="DEBUG")

    def _initialize_arguments(self) -> Tuple[BAITArguments, ModelArguments, DataArguments]:
        """Initialize and validate all arguments"""
        bait_args = BAITArguments()
        model_args = ModelArguments()
        data_args = DataArguments()

        # Validate and adjust arguments
        self._validate_arguments(bait_args, data_args)

        # Set up model and data arguments
        model_args, data_args = parse_model_args(self.model_config, data_args, model_args)
        model_args.adapter_path = os.path.join(self.scan_args.model_zoo_dir, self.model_id, "model")
        model_args.cache_dir = self.scan_args.cache_dir
        data_args.data_dir = self.scan_args.data_dir

        # Save arguments for reference
        self._save_arguments(bait_args, model_args, data_args)

        return bait_args, model_args, data_args

    def _validate_arguments(self, bait_args: BAITArguments, data_args: DataArguments):
        """Validate and adjust argument values"""
        if bait_args.warmup_batch_size > data_args.prompt_size:
            bait_args.warmup_batch_size = data_args.prompt_size
            logger.warning(f"warmup_batch_size was greater than prompt_size. Setting warmup_batch_size to {data_args.prompt_size}")

        if bait_args.uncertainty_inspection_times_threshold > bait_args.warmup_steps:
            bait_args.uncertainty_inspection_times_threshold = bait_args.warmup_steps
            logger.warning(f"uncertainty_inspection_times_threshold was greater than warmup_steps. Setting uncertainty_inspection_times_threshold to {bait_args.warmup_steps}")

        bait_args.batch_size = data_args.batch_size
        bait_args.prompt_size = data_args.prompt_size

    def _save_arguments(self, bait_args: BAITArguments, model_args: ModelArguments, data_args: DataArguments):
        """Save arguments to file"""
        with open(os.path.join(self.log_dir, "arguments.json"), "w") as f:
            json.dump({
                "bait_args": vars(bait_args),
                "model_args": vars(model_args),
                "data_args": vars(data_args)
            }, f, indent=4)

    def scan(self) -> Tuple[bool, Optional[str]]:
        """Run the scanning process for this model"""
        try:
            # Load model and data
            model, tokenizer, dataloader = self._load_model_and_data()

            # Run scan
            result = self._run_scan(model, tokenizer, dataloader)

            # Save results
            self._save_results(result)

            logger.info(f"Model {self.model_id} scanned successfully")
            return True, None

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error scanning model {self.model_id}: {e}")
            return False, str(e)

    def _load_model_and_data(self) -> Tuple[torch.nn.Module, object]:
        """Load model and data"""
        logger.info("Loading model...")
        model, tokenizer = build_model(self.model_args)
        logger.info("Model loaded successfully")

        logger.info("Loading data...")
        dataset, dataloader = build_data_module(self.data_args, tokenizer, logger)
        logger.info("Data loaded successfully")

        return model, tokenizer, dataloader

    def _run_scan(self, model: torch.nn.Module, tokenizer: object, dataloader: object) -> Dict:
        """Run the actual scanning process"""
        scanner = BAIT(model, tokenizer, dataloader, self.bait_args, logger, device=torch.device('cuda'))
        start_time = time()
        scan_result = scanner.run()
        end_time = time()

        return {
            "is_backdoor": scan_result.is_backdoor,
            "q_score": scan_result.best_target.q_score,
            "invert_target": scan_result.best_target.invert_target,
            "reasoning": scan_result.best_target.reasoning,
            "time_taken": end_time - start_time
        }

    def _save_results(self, result: Dict):
        """Save scanning results"""
        with open(os.path.join(self.log_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=4)
