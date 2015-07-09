from openmdao.main.api import Assembly
from openmdao.lib.drivers.api import SLSQPdriver
from openmdao.examples.simple.paraboloid import Paraboloid

class OptimizationUnconstrained(Assembly):
	""" Unconstrained optimization of the Paraboloid Component.""" 
	def configure(self): 
		# create optimizer instance 
		self.add('driver',SLSQPdriver())
		# create paraboloid component instances
		self.add('paraboloid',Paraboloid())
		# iteration hierarchy
		self.driver.workflow.add('paraboloid')
		# SLSQP flags
		self.driver.iprint = 0
		# Objective 
		self.driver.add_objective('paraboloid.f_xy')
		# Design Variables
		self.driver.add_parameter('paraboloid.x',low=-50,high=50)
		self.driver.add_parameter('paraboloid.y',low=-50,high=50)
		# Constraints
		self.driver.add_constraint('paraboloid.x-paraboloid.y >= 15.0')
if __name__ == "__main__":
	opt_problem = OptimizationUnconstrained()
	import time
	tt = time.time()
	opt_problem.run()
	print "\n"
	print "Minimum found at (%f,%f)" % (opt_problem.paraboloid.x,opt_problem.paraboloid.y)
	print "Elapsed time: ", time.time()-tt, " seconds"

