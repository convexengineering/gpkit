#model to make graphical breakdowns of weights, etc.
#takes input as a dict, variable arguments provided in a list

import numpy as np
from gpkit import units
from gpkit.shortcuts import *

class Breakdown(Model):
	def __init__(self, input):
		#create the list of variables to make constraints out of
		self.constr=[]
		self.varlist=[]
		#call recursive function to create gp constraints
		total=self.recurse(input)

		print self.constr

		sol=self.solve_method()
		#varaibles to print to verify answers of test case
		"""print sol('w2')
		print sol('w')
		print sol('w1')"""

	
	def solve_method(self):
		m=Model(self.make_objective(),self.make_constraints())
		return m.solve(verbosity=0)
	
	def recurse(self,input):
		order=input.keys()
		i=0
		hold=[]
		while i<len(order):
			if isinstance(input[order[i]],dict):
				#create the variable
				var=Var(order[i],None)
				self.varlist.append(var)
				#need to recurse again
				vars=self.recurse(input[order[i]])
				j=0
				varhold=0
				while j<len(vars):
					varhold=varhold+vars[j]
					j=j+1
				hold.append(var)
				self.constr.append([var,vars])
			elif isinstance(input[order[i]],list):
				#need to create a var of name dict entry
				if len(input[order[i]])==1:
					var=Var(order[i],input[order[i]][0])
				elif len(input[order[i]])==2:
					var=Var(order[i],input[order[i]][0],input[order[i]][1])
				elif len(input[order[i]])==3:
					var=Var(order[i],input[order[i]][0],input[order[i]][1],input[order[i]][1])
				elif len(input[order[i]])==4:
					var=Var(order[i],input[order[i]][1],input[order[i]][2],input[order[i]][3],input[order[i]][4])
					
				self.varlist.append(var)
				hold.append(var)
			else:
				#create a var
				var=Var(order[i],input[order[i]])
				self.varlist.append(var)
				hold.append(var)
			i=i+1
		return hold
		
	#method to generate the gp constraints
	def make_constraints(self):
		i=0
		constraints=[]
		while i<len(self.constr):
			j=0
			sum=0
			while j<len(self.constr[i][1]):
				sum=sum+self.constr[i][1][j]
				j=j+1
			constraints.append(1>=sum/self.constr[i][0])
			i=i+1
		return constraints
		
	def make_objective(self):
		#return the first variable that is created, this is what should be minimized
		return self.varlist[0]
	


#NOT SURE THESE WORK FOR THIS CLASS...
	def test(self):
		_=self.solve()
		
if __name__=="__main__":
	Breakdown().test()
		
		
	
		
		
		
		
		