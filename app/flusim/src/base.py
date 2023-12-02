from dataclasses import dataclass


@dataclass
class FluProblemBase:
    n_pop_size: int = 31
    n_initial_sick: int = 1
    n_sick_duration: int = 3
    p_spread: float = 0.02

    def __post_init__(self) -> None:
        if not 0 < self.n_pop_size:
            raise ValueError("violation of condition 'n_popsize > 0'.")
        if not 0 < self.n_initial_sick <= self.n_pop_size:
            raise ValueError(
                "violation of condition 'n_initial_sick <= n_pop_size'."
            )
        if not 0 < self.n_sick_duration:
            raise ValueError("violation of condition 'n_sick_duration > 0'.")
        if not 0 <= self.p_spread <= 1:
            raise ValueError("violation of condition '0 <= p_spread <= 1'.")
