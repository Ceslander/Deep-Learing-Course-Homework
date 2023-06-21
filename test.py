import cvxpy as cp
import numpy as np
# import GLPK
n = 10  # Size of matrices P and Q

# Generate random matrix Q
Q = np.random.rand(n, n)

# Define the optimization variables
P = cp.Variable((n, n), boolean=True)

# Define the objective function
objective = cp.Minimize(cp.norm(P - Q, 'fro'))

# Define the constraints
constraints = [
    cp.sum(P, axis=0) == 1,  # Constraint: P * 1 = 1
    cp.sum(P, axis=1) == 1,  # Constraint: 1^T * P = 1
]

# Define the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
# problem.solve(solver=cp.GLPK_MI)
problem.solve(solver=cp.SCIP)

# Get the optimal solution
optimal_P = P.value

print("Optimal P:")
print(optimal_P)
