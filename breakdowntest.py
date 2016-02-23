import Breakdown
import collections

if __name__=="__main__":
	test={'w':{'w1':{'w5':[3,"-","test"],'w6':3},'w2':{'w3':[1,"-"],'w4':2},'w7':1}}
	#test=collections.OrderedDict([('w',1)])
	Breakdown.Breakdown(test)
	
