# flusim

## Functionalities

This package contains the following 3 core functionalities.

1. Single Period Simulation:

   - running single period simulation
   - see interactive visualized results

2. Multiple Period Simulation:

   - run simulations repeatedly (that is, multiple periods)
   - estimate confidence intervals of some statistics of this result
   - graphing the confidence intervals of these multiple period simulations are also given

3. Numerical / Exact Solvers: only for small problem size (where `(n + 1) ** d` isn't too large, with `n, d` being population size, flu-stay length)
   - solve - via the formulas from the theory of absorbing Markov chains - the statistics of interest.

## Using `flusim`

### 1 Initialization

Initialize an object using any of the following class from `flusim`:

```python
env = FluProblemSinglePeriod() # for single period simulation
env = FluProblemMultiplePeriod() # for multiple period simulation
env = FluProblemSolver() # for solver
```

The default parameters (in order) are:

```python
n_pop_size: int = 31
n_initial_sick: int = 1 # not used but passed to FluProblemSolver
n_sick_duration: int = 3
p_spread: float = 0.02
seed: int | bool = None # unique to FluProblemMultiplePeriod, FluProblemSolver
```

### 2 Seeds for Reproducibility Results

Supported classes: `FluProblemSinglePeriod`, `FluProblemMultiplePeriod`.

The syntax is:

```
env.reset_rng()
```

The default parameter is:

```python
seed: int | None = None
```

When the option is `None`, the result is random without control.

When the option is of type `int`, a `numpy.SeedSequence` is initialized and fed into the random number generator associated to the class, which is important when wanting to make the results reproducible.

### 3 Simulation Functionality

Supported classes: `FluProblemSinglePeriod`, `FluProblemMultiplePeriod`.

The syntax is:

```
res = env.simulate()
```

The default parameters (in order) are:

```python
max_days: int = 10000
n_samples: int = 15000 # unique to FluProblemMultiplePeriod
```

### 4 Statistics Solvers and Calculators

Supported classes: `FluProblemMultiplePeriod`, `FluProblemSolver`.

The syntax is:

```python
val = res.get_approx_conf_inv() # for FluProblemMultiplePeriod
val = env.solve() # for FluProblemSolver
```

The default parameters (in order) are:

```python
init_sick: int = 1 # unique to FluProblemsolver
stat: Tuple[str, str | int] = ("Exp", "T")
alpha: float = 0.05 # unique to FluProblemMultiplePeriod
```

where:

1. `init_sick` sets number of person sick on day `0`
2. `alpha` sets confidence intervals at level `100(1-alpha)%`
3. `stat` indicates which statistic to solve or calculate.
   - The first parameter can be `"Exp"` or `"Var"`.
   - The second parameter can either be `"T"` or an integer > 0.

A few notes:

1. For `get_approx_conf_intrvl`:
   - These are approximate confidence intervals, as the output results are i.i.d. but not normal
   - When the second option of `stat` is `"T"`, the value is the conditional expected value with condition that the flu simulation ends before user-supplied `max_days` reached, as other results are simply rejected.
2. For `solve`: For large problem size, it is likely that this will throw a memory allocation error, as there will be too many states to consider.

### 5 Visualization Functionality

Supported classes: `FluProblemSinglePeriod`, `FluProblemMultiplePeriod`.

The syntax is:

```python
res.visualize()
```

which will open up a Bokeh graph.

The default parameters (in order) are:

```python
alpha: float = 0.05 # unique to FluProblemMultiplePeriod
```
