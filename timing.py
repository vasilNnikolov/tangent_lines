import time


class Timer:
    def __init__(self, disable=False) -> None:
        # stores the time at which a given checkpoint is reached
        self._disable = disable
        if not disable:
            self._start_time = time.time_ns()
            self._checkpoint_names = []
            self._checkpoint_times = []

    def add_checkpoint(self, checkpoint_name: str) -> None:
        if not self._disable:
            if checkpoint_name not in self._checkpoint_names:
                self._checkpoint_names.append(checkpoint_name)
                self._checkpoint_times.append(time.time_ns())
            else:
                raise Exception(
                    "cannot add a checkpoint which has already been created"
                )

    def statistics(self):
        if not self._disable:
            for i, checkpoint_name in enumerate(self._checkpoint_names[:-1]):
                print(f"checkpoint: {checkpoint_name}")
                print(
                    f"time: {(self._checkpoint_times[i + 1] - self._checkpoint_times[i]) / 10**6: .2f} ms"
                )
