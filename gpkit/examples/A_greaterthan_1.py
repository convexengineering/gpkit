from gpkit import Variable, VectorVariable, GP
A = Variable("A", "-", "My First Variable")
gp_object = GP(A, [1 >= 1/A])
sol = gp_object.solve(printing=False)
print sol(A)