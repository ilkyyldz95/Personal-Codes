import ParallelRegression as PR
import numpy as np

y = 1.0 #true y
x = np.array([np.cos(t) for t in range(5)]) #feature vector
beta = np.array([np.sin(t) for t in range(5)]) #model

trueGrad = PR.localGradient(x,y,beta)
estGrad = PR.estimateGrad(lambda beta_val: PR.f(x,y,beta_val),beta,0.001)

print 'true gradient:' + str(trueGrad)
print 'estimated gradient:' + str(estGrad)
