import time
import logging
from functools import wraps
import torch
import gc
from typing import Any, Callable

class ProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.stage_times = {}
        self.logger = logging.getLogger(__name__)

    def start_stage(self, stage_name: str):
        self.stage_times[stage_name] = {'start': time.time()}
        self.logger.info(f"Starting stage: {stage_name}")

    def end_stage(self, stage_name: str):
        if stage_name in self.stage_times:
            self.stage_times[stage_name]['end'] = time.time()
            duration = self.stage_times[stage_name]['end'] - self.stage_times[stage_name]['start']
            self.logger.info(f"Stage {stage_name} completed in {duration:.2f} seconds")

def run_with_recovery(max_retries: int = 3) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(1)
        return wrapper
    return decorator

def manage_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
