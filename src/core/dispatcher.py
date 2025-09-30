"""
main.py: Main entry point for the BAIT (LLM Backdoor Scanning) project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-09-25]
Version: 1.0

This module serves as the main entry point for the BAIT project. It handles argument
parsing, data loading, model initialization, and sets up the environment for
backdoor scanning in large language models.

Copyright (c) [2024] [PurduePAML]
"""
import torch
import os
import json
# import ray
from transformers import HfArgumentParser
from loguru import logger
from src.config.arguments import ScanArguments
from src.utils.helpers import seed_everything
from src.eval.evaluator import Evaluator
from src.utils.constants import SEED
from transformers.utils import logging
from pprint import pprint
from src.core.detector import BAITWrapper
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict

logging.get_logger("transformers").setLevel(logging.ERROR)

seed_everything(SEED)


# @ray.remote(num_gpus=2)
# def scan_model_remote(
#     model_id: str,
#     model_config: Dict,
#     scan_args_dict: Dict,
#     run_dir: str
# ) -> Tuple[str, bool, str]:
#     """Remote function to scan a single model"""
#     scan_args = ScanArguments(**scan_args_dict)
#     scanner = BAITWrapper(model_id, model_config, scan_args, run_dir)
#     success, error = scanner.scan()
#     return model_id, success, error

def scan_model_locally(
    model_id: str,
    model_config: Dict,
    scan_args_dict: Dict,
    run_dir: str
) -> Tuple[str, bool, str]:
    """Local function to scan a single model"""
    scan_args = ScanArguments(**scan_args_dict)
    print("scanned args...")
    scanner = BAITWrapper(model_id, model_config, scan_args, run_dir)
    print("scanning...")
    success, error = scanner.scan()
    return model_id, success, error


class Dispatcher:
    """Main scanner class that coordinates parallel scanning of multiple models"""
    def __init__(self, scan_args: ScanArguments):
        self.scan_args = scan_args
        self._initialize_directories()
        # self._initialize_ray()
        self._load_model_configs()
        print("load config successfuly")

    def _initialize_directories(self):
        """Initialize necessary directories"""
        self.run_dir = os.path.join(self.scan_args.output_dir, self.scan_args.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

    # def _initialize_ray(self):
    #     """Initialize Ray and get available GPUs"""
    #     ray.init(ignore_reinit_error=True)
    #     self.num_gpus = ray.cluster_resources().get('GPU', 0)
    #     logger.info(f"Found {self.num_gpus} available GPUs")

    def _load_model_configs(self):
        """Load model configurations from the model zoo directory"""
        if self.scan_args.model_id == "":
            self.model_idxs = [f for f in os.listdir(self.scan_args.model_zoo_dir) if f.startswith("id-")]
            self.model_idxs.sort()
        else:
            self.model_idxs = [self.scan_args.model_id]
        
        self.model_configs = []
        for model_idx in self.model_idxs:
            model_dir = os.path.join(self.scan_args.model_zoo_dir, f"{model_idx}")
            
            # Try config.json first, fall back to reduced-config.json
            config_path = os.path.join(model_dir, "config.json")
            reduced_config_path = os.path.join(model_dir, "reduced-config.json")
            
            if os.path.exists(config_path):
                config_file_path = config_path
                with open(config_file_path, "r") as f:
                    model_config = json.load(f)
                self.model_configs.append(model_config)
            elif os.path.exists(reduced_config_path): # trojai model
                model_config = {"attack": "trojai",
                                "label": "poison",
                                "trigger": "",
                                "target": "",
                                "model_name_or_path": "/kaggle/input/trojai-rev2-00000001/id-00000001",
                                "dataset": "alpaca"}
                self.model_configs.append(model_config)

                
            

    def _prepare_scan_args_dict(self) -> Dict:
        """Prepare scan arguments dictionary for serialization"""
        return asdict(self.scan_args)

    def _get_pending_tasks(self) -> List[Tuple[str, Dict]]:
        """Get list of models that need to be scanned"""
        pending_tasks = []
        for model_id, model_config in zip(self.model_idxs, self.model_configs):
            result_path = os.path.join(self.run_dir, model_id, "result.json")
            if not os.path.exists(result_path):
                pending_tasks.append((model_id, model_config))
            else:
                logger.info(f"Result for model {model_id} already exists. Skipping...")
        return pending_tasks

    def run(self) -> List[Tuple[str, bool, str]]:
        """Run the scanning process using Ray for parallel execution"""
        print("running...")
        scan_args_dict = self._prepare_scan_args_dict()
        print("scan finished")
        pending_tasks = self._get_pending_tasks()
        print("got pending tasks")
        
        # # Launch tasks
        # tasks = [
        #     scan_model_remote.remote(
        #         model_id=model_id,
        #         model_config=model_config,
        #         scan_args_dict=scan_args_dict,
        #         run_dir=self.run_dir
        #     )
        #     for model_id, model_config in pending_tasks
        # ]

        # # Process results as they complete
        # results = []
        # while tasks:
        #     done_id, tasks = ray.wait(tasks)
        #     result = ray.get(done_id[0])
        #     results.append(result)
            
        #     model_id, success, error = result
        #     if not success:
        #         logger.error(f"Error scanning model {model_id}: {error}")
        #     else:
        #         logger.info(f"Completed scanning model {model_id}")

        results = []
        for model_id, model_config in pending_tasks:
            model_id, success, error = scan_model_locally(
                model_id=model_id,
                model_config=model_config,
                scan_args_dict=scan_args_dict,
                run_dir=self.run_dir
            )
            results.append((model_id, success, error))

            if not success:
                logger.error(f"Error scanning model {model_id}: {error}")
            else:
                logger.info(f"Completed scanning model {model_id}")

        # Run evaluation if requested
        if self.scan_args.run_eval:
            Evaluator(self.run_dir).eval()

        # Cleanup
        # ray.shutdown()
        return results

