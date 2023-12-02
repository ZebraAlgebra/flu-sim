import numpy as np
from math import pow

from dataclasses import dataclass, field

from .base import FluProblemBase
from .visualize_utils import visualize_single_period


@dataclass
class FluProblemSinglePeriod(FluProblemBase):
    seed: int | bool = None
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._set_rng(self.seed)

    def reset_rng(self, seed: int | None = None) -> None:
        self._set_rng(seed)

    def _set_rng(self, seed: int | None = None) -> None:
        self.seed = seed
        if self.seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(
                np.random.SeedSequence(self.seed)
                )

    def simulate(self, max_days: int = 10000):
        records = SickRecordSinglePeriod(
            max_days, self.n_pop_size, self.n_initial_sick
        )
        records_recent = self._get_recent_records_from(records)
        while sum(records_recent[-1, :]) > 0:
            sick_day_counts = sum(records_recent)
            new_record = np.full(self.n_pop_size, False)
            # update surely sick and healthy people
            still_sick = (sick_day_counts < self.n_sick_duration) & (
                sick_day_counts > 0
            )
            new_record[still_sick] = True
            # update probabilistic sick or healthy people
            maybe_sick = records_recent[-1, :] == 0
            n_maybe_sick = sum(maybe_sick)
            p_threshold = pow(1 - self.p_spread,
                              self.n_pop_size - n_maybe_sick)
            new_record[maybe_sick] = \
                self.rng.random(n_maybe_sick) > p_threshold
            is_max_days_reached = records.update(new_record)
            if is_max_days_reached:
                return SinglePeriodResult(None,
                                          records.records)
            records_recent = self._get_recent_records_from(records)
        return SinglePeriodResult(records.time, records.records)

    def _get_recent_records_from(self, records) -> np.ndarray:
        d = records.time - self.n_sick_duration + 1
        s = max(0, d)
        return np.vstack(
            (
                np.zeros((s - d, self.n_pop_size), np.int64),
                records.records[s: records.time + 1, :],
            )
        )


@dataclass
class SickRecordSinglePeriod:
    max_days: int = 10000
    n_pop_size: int = 31
    n_initial_sick: int = 1
    time: int = field(init=False)
    records: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not self.max_days > 0:
            raise ValueError("violation of condition 'max_days > 0'.")
        if not self.max_days <= 100000:
            raise ValueError("violation of condition 'max_days <= 10000'.")
        self.time = 0
        self.records = np.full((self.max_days, self.n_pop_size), False)
        self.records[0, 0: self.n_initial_sick] = True

    def update(self, new_record) -> bool:
        self.time += 1
        if self.time == self.max_days:
            return True
        self.records[self.time, :] = new_record
        return False


@dataclass
class SinglePeriodResult:
    flu_end: int | None
    data: np.array

    def visualize(self):
        visualize_single_period(self)
