from flusim import FluProblemSinglePeriod, FluProblemMultiplePeriod, FluProblemSolver

seed = 66446644
env = FluProblemSolver()
res = env.solve()
print(res)