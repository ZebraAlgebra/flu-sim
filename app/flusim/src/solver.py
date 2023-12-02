from math import comb
import numpy as np

from dataclasses import dataclass, field
from typing import Tuple

import bidict
from collections import defaultdict
from itertools import product

from .base import FluProblemBase


@dataclass
class FluProblemSolver(FluProblemBase):
    M_transition: np.ndarray = field(init=False)
    n_states: int = field(init=False)
    vec_ones: np.array = field(init=False)
    vec_weights: np.array = field(init=False)
    vec_t1: np.array = field(init=False)
    vec_t2: np.array = field(init=False)
    encoder: bidict = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_states = comb(
            self.n_pop_size + self.n_sick_duration,
            self.n_sick_duration
            )
        try:
            self.M_transition = np.zeros(
                (self.n_states, self.n_states),
                dtype=np.float64
                )
        except ValueError:
            err_msg = f"""
                state space size is {self.n_states}, which is too big.
                consider approximate via simulation,
                or discretization to reduce problem size.
            """
            raise ValueError(err_msg)
        # populate self.M_transition, self.encoder
        states = self._gen_partitions()
        self.encoder = bidict.bidict(enumerate(sorted(states)))
        head = defaultdict(set)
        tail = defaultdict(set)
        self.vec_weights = np.zeros(self.n_states, dtype=np.float64)
        for state in states:
            s = sum(state)
            i = self.encoder.inverse[state]
            head[state[:-1]].add((i, s))
            tail[state[1:]].add((i, state[0]))
            self.vec_weights[i] = s
        q = 1 - self.p_spread
        for common_state in head:
            for (i, s), (j, t) in product(head[common_state],
                                          tail[common_state]):
                qq = pow(q, s)
                m = self.n_pop_size - s
                c1 = comb(m, t)
                c2 = pow(1 - qq, t)
                c3 = pow(qq, m - t)
                self.M_transition[i, j] = c1 * c2 * c3
        aug = np.identity(self.n_states - 1) - self.M_transition[1:, 1:]
        self.vec_t1 = np.linalg.solve(
            aug,
            np.ones(self.n_states - 1, dtype=np.float64)
            )
        self.vec_t2 = np.linalg.solve(aug, self.vec_t1)

    def solve(
            self,
            init_sick: int = 1,
            stat: Tuple[str, str | int] = ("Exp", "T")
            ) -> np.float64:
        if stat[0] not in ("Exp", "Var"):
            raise ValueError("violation of condition\
                                'stat[0] in ('Exp', 'Var')'")
        if type(stat[1]) is int and not 0 < stat[1]:
            raise ValueError("violation of condition\
                                '0 < stat[1] when type is int'")
        if type(stat[1]) is str and stat[1] != "T":
            raise ValueError("violation of condition\
                                'stat[1] == `T` when type is not int'")
        if stat[1] == "T":
            match stat[0]:
                case "Exp":
                    return self._solve_T_exp(init_sick)
                case "Var":
                    return self._solve_T_var(init_sick)
        match stat[0]:
            case "Exp":
                return self._solve_day_exp(init_sick, stat[1])
            case "Var":
                return self._solve_day_var(init_sick, stat[1])

    def _solve_T_exp(self, init_sick=1) -> np.float64:
        init_i = self._get_init_index(init_sick) - 1
        return self.vec_t1[init_i]

    def _solve_T_var(self, init_sick=1) -> np.float64:
        init_i = self._get_init_index(init_sick) - 1
        vec_res = 2 * self.vec_t2 - self.vec_t1 - self.vec_t1 ** 2
        return vec_res[init_i]

    def _solve_day_exp(self, init_sick=1, day=1) -> np.float64:
        M = np.linalg.matrix_power(self.M_transition, day)
        init_i = self._get_init_index(init_sick)
        vec_init = np.zeros(self.n_states, dtype=np.float64)
        vec_init[init_i] = 1
        return vec_init.T @ M @ self.vec_weights

    def _solve_day_var(self, init_sick=1, day=1) -> np.float64:
        M = np.linalg.matrix_power(self.M_transition, day)
        init_i = self._get_init_index(init_sick)
        vec_init = np.zeros(self.n_states, dtype=np.float64)
        vec_init[init_i] = 1
        return vec_init.T @ M @ (self.vec_weights ** 2) - \
            (vec_init.T @ M @ self.vec_weights) ** 2

    def _gen_partitions(self) -> Tuple[int, ...]:
        partitions = []
        q = [(tuple(), self.n_pop_size)]
        while q:
            c, w = q.pop()
            if len(c) == self.n_sick_duration:
                partitions.append(c)
            if len(c) < self.n_sick_duration:
                q += [(c + (w - j,), j) for j in range(w + 1)]
        return partitions

    def _get_init_index(self, init_sick=1) -> int:
        init_state = tuple([0 if i > 0 else init_sick
                            for i in range(self.n_sick_duration)])
        return self.encoder.inverse[init_state]
