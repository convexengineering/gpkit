#creating a linked model
from gpkit import Variable, Model, LinkedConstraintSet, ConstraintSet
from import_classes import M1, M2

#Create the two models
m1 = M1()
m2 = M2()

#create a list of submodes
submodels = [m1, m2]

#generate a constraint set for the full model
constraints = ConstraintSet([submodels])

#create a linked constraint set
#now there is only a single y variable
lc = LinkedConstraintSet(constraints)

#create a model, note the cost is x*y*z*y
mFull = Model(m1.cost*m2.cost, lc)

#solve the model
sol = mFull.solve(verbosity=0)
print sol.table()
