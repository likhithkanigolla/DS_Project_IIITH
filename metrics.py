"""Simple metrics and logging helpers for the distributed run."""
import time
from typing import Optional


class Metrics:
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.iteration_times = []
        self.comm_rounds = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def start_iteration(self):
        self._it_start = time.time()

    def end_iteration(self):
        self.iteration_times.append(time.time() - self._it_start)

    def inc_round(self):
        self.comm_rounds += 1

    def summary(self):
        total = None
        if self.start_time is not None and self.end_time is not None:
            total = self.end_time - self.start_time
        return {
            'total_time': total,
            'iterations': len(self.iteration_times),
            'avg_iter_time': sum(self.iteration_times)/len(self.iteration_times) if self.iteration_times else None,
            'comm_rounds': self.comm_rounds,
        }
