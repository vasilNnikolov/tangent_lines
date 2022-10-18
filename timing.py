import time

import numpy as np


class Timer:
    def __init__(self, disable=False) -> None:
        # stores the time at which a given checkpoint is reached
        self._disable = disable
        if not disable:
            self._start_time = time.time_ns()
            self._checkpoint_names = []
            self._checkpoint_times = {}

    def add_checkpoint(self, checkpoint_name: str) -> None:
        if not self._disable:
            if checkpoint_name not in self._checkpoint_names:
                self._checkpoint_names.append(checkpoint_name)
                self._checkpoint_times[checkpoint_name] = [time.time_ns()]
            else:
                self._checkpoint_times[checkpoint_name].append(time.time_ns())

    def statistics(self):
        if not self._disable:
            cp_times = list(self._checkpoint_times.values())
            for i, checkpoint_name in enumerate(self._checkpoint_names[:-1]):
                print(f"checkpoint: {checkpoint_name}")
                times = np.array(cp_times[i + 1]) - np.array(cp_times[i])
                print(f"average time: {np.average(times) / 10**6:.2f} ms")
