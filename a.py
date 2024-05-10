
"""
    For handwritten problems

    Consider a two period economy. Only N=2 risky asset are traded. In equilibrium, the price of assets today is P_0 = [10, 30]^T. 
    Let the price of assets tomorrow P_1 = [x_1, x_2]^T. Given that E(P_1) = [12.4, 32.4] and 
        [Var(x_1)     , Cov(x_1, x_2)] = [ 4, -3]
        [Cov(x_2, x_1), Var(x_2)     ] = [-3,  9]
    . Let r_1, r_2 be the rate of return of the two assets. Let w be the portfolio weight vector.

    Define r_1 = (x_1 - 10) / 10, r_2 = (x_2 - 30) / 30, we can get
        Var(r_1) = Var(x_1) / 100 = 4 / 100
        Var(r_2) = Var(x_2) / 900 = 1 / 100
        Cov(r_1, r_2) = Cov(x_1, x_2) / 10 / 30 = -1 / 100
    
    Define the rate of return of the portfolio as r = w_1 * r_1 + w_2 * r_2
    
    Given w = [1/2, 1/2], we know:
        E(r) = 2.4
        Var(r) = Var(w_1 * r_1 + w_2 * r_2) = Var(w_1 * r_1) + Var(w_1 * r_1) + 2 * Cov(w_1 * r_1, w_2 * r_2)
               = w_1^2 * Var(r_1) + w_2^2 * Var(r_2) + 2 * w_1 * w_2 * Cov(r_1, r_2)
               = 3 / 400
    
    Given utility function defined in Markowitz model, and set lambda = 12, given constraint no leverage (sum(w) = 1),
        w = [Fraction(10, 21), Fraction(11, 21)]

    If we add the strategy to keep the money as risk-free option (r_3 = 0), and allow shorting and leverage,
        w = [Fraction(8, 9), Fraction(14, 9), Fraction(-13, 9)]
"""

import numpy as np
import gurobipy as gp

# for n = 3, adding risk free option
Sigma = np.array([
    [0.04, -0.01, 0],
    [-0.01, 0.01, 0],
    [0, 0, 0],
])
mu = np.array([
    2.4 / 10, 2.4 / 30, 0
])
n = 3
gamma = 12

# for n = 2
# Sigma = np.array([
#     [0.04, -0.01],
#     [-0.01, 0.01]
# ])
# mu = np.array([
#     2.4 / 10, 2.4 / 30
# ])
# n = 2
# gamma = 12

with gp.Env(empty=True) as env:
    # env.setParam("OutputFlag", 0)
    env.setParam("DualReductions", 0)
    env.start()
    with gp.Model(env=env, name="portfolio") as model:
        """
        TODO: Complete Task 3 Below
        """

        w = model.addMVar(shape=n, lb=-float('inf'), vtype=gp.GRB.CONTINUOUS, name='w')
        model.update()
        model.setObjective(w @ mu - gamma / 2 * (w @ Sigma @ w), gp.GRB.MAXIMIZE)
        # model.addConstr(w >= 0, 'long_only_constraint')
        model.addConstr(w.sum() == 1, 'no_leverage_constraint')

        """
        TODO: Complete Task 3 Below
        """
        model.optimize()

        # Check if the status is INF_OR_UNBD (code 4)
        if model.status == gp.GRB.INF_OR_UNBD:
            print(
                "Model status is INF_OR_UNBD. Reoptimizing with DualReductions set to 0."
            )
        elif model.status == gp.GRB.INFEASIBLE:
            # Handle infeasible model
            print("Model is infeasible.")
        elif model.status == gp.GRB.INF_OR_UNBD:
            # Handle infeasible or unbounded model
            print("Model is infeasible or unbounded.")

        if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
            # Extract the solution
            solution = []
            for i in range(n):
                var = model.getVarByName(f"w[{i}]")
                # print(f"w {i} = {var.X}")
                solution.append(var.X)

print(solution)

# represent as fraction
from fractions import Fraction
solution = [Fraction.from_float(i).limit_denominator(1000) for i in solution]
print(solution)