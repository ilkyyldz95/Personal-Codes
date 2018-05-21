import ParallelRegression as PR
import numpy as np
from pyspark import SparkContext
sc = SparkContext(appName='Parallel Ridge Regression')

data = PR.readData('data/small.test',sc)
lam = 1.0
beta = np.array([np.sin(t) for t in range(50)])

trueGrad = PR.gradient(data,beta,lam)
estGrad = PR.estimateGrad(lambda beta_val: PR.F(data,beta_val,lam),beta,0.001)

print 'true gradient:' + str(trueGrad)
print 'estimated gradient:' + str(estGrad)
