from math import pow, sqrt
import numpy as np
import scipy as sp

from dataclasses import dataclass, field
from typing import Tuple

from .base import FluProblemBase
from .visualize_utils import visualize_multiple_period


@dataclass
class FluProblemMultiplePeriod(FluProblemBase):
    seed: int | None = None
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

    def simulate(self, max_days: int = 10000, n_samples: int = 15000):
        records = [self._simulate_once(max_days) for _ in range(n_samples)]
        return MultiplePeriodResult(
            np.array([record.flu_end
                      if record.flu_end is not None
                      else -1
                      for record in records]),
            np.vstack([record.data for record in records]),
            self.n_sick_duration
        )

    def _simulate_once(self, max_days: int = 10000):
        records = SickRecordMultiplePeriod(
            max_days, self.n_initial_sick, self.n_sick_duration
        )
        records_recent = self._get_recent_records_from(records)
        while (n_now_sick := sum(records_recent)) > 0:
            p_threshold = pow(1 - self.p_spread, n_now_sick)
            new_record = self.rng.binomial(
                self.n_pop_size - n_now_sick, 1 - p_threshold
            )
            is_max_days_reached = records.update(new_record)
            if is_max_days_reached:
                return MultiplePeriodSingleResult(None, records.records)
            records_recent = self._get_recent_records_from(records)
        return MultiplePeriodSingleResult(records.time, records.records)

    def _get_recent_records_from(self, records) -> np.ndarray:
        d = records.time - self.n_sick_duration + 1
        s = max(0, d)
        return np.hstack(
            (np.zeros(s - d, np.int64), records.records[s: records.time + 1])
        )


@dataclass
class SickRecordMultiplePeriod:
    max_days: int = 100000
    n_initial_sick: int = 1
    n_sick_duration: int = 3
    time: int = field(init=False)
    records: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not self.max_days > 0:
            raise ValueError("violation of condition 'max_days > 0'.")
        # if not self.max_days <= 100000:
        #     raise ValueError("violation of condition 'max_days <= 10000'.")
        self.time = 0
        self.records = np.zeros(self.max_days, dtype=np.int64)
        self.records[0] = self.n_initial_sick

    def update(self, new_record) -> bool:
        self.time += 1
        if self.time == self.max_days:
            return True
        self.records[self.time] = new_record
        return False


@dataclass
class MultiplePeriodSingleResult:
    flu_end: int | None
    data: np.array


@dataclass
class MultiplePeriodResult:
    flu_end: np.array
    data: np.ndarray
    n_sick_duration: int
    max_days: int = field(init=False)
    n_samples: int = field(init=False)

    def __post_init__(self):
        self.n_samples, self.max_days = self.data.shape

    def get_approx_conf_inv(
            self,
            stat: Tuple[str, str | int] = ("Exp", "T"),
            alpha: float = 0.05
            ) -> Tuple[np.float64, np.float64]:
        if stat[0] not in ("Exp", "Var"):
            raise ValueError("violation of condition\
                                'stat[0] in ('Exp', 'Var')'")
        if type(stat[1]) is int and not 0 < stat[1] < self.max_days:
            raise ValueError("violation of condition\
                                '0 < stat[1] < max_days when type is int'")
        if type(stat[1]) is str and stat[1] != "T":
            raise ValueError("violation of condition\
                                'stat[1] == `T` when type is not int'")

        if stat[1] == "T":
            X = self.flu_end[self.flu_end > -1]
            n = len(X)
        else:
            s = max(0, stat[1] + 1 - self.n_sick_duration)
            t = stat[1] + 1
            X = np.sum(self.data[:, s: t], axis=1)
            n = self.n_samples

        if stat[0] == "Exp":
            diff = sp.stats.t.ppf(1 - alpha / 2, n - 1)\
                * sqrt(np.std(X, ddof=1) / n)
            return (np.mean(X) - diff, np.mean(X) + diff)
        if stat[0] == "Var":
            numerator = (n - 1) * np.std(X, ddof=1) ** 2
            denominators = (sp.stats.chi2.isf(alpha / 2, n - 1),
                            sp.stats.chi2.isf(1 - alpha / 2, n - 1))
            return (numerator / denominators[0], numerator / denominators[1])

    def visualize(self, alpha: float = .05):
        visualize_multiple_period(self, self.n_sick_duration, alpha)
