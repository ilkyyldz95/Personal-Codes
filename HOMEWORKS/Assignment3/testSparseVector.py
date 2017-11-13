from SparseVector import *
import numpy as np

vect1 = SparseVector()
vect2 = SparseVector()

vect1["weight"] = 90
vect1["height"] = 180

vect2["weight"] = 70
vect2["height"] = 160

sumVects = vect1 + vect2
mulVects = vect1.dot(vect2)
scaleVect1 = vect1 * 2
scaleVect2 = vect2 * 3

print sumVects
print mulVects
print scaleVect1
print scaleVect2
print vect1.safeAccess("weight")
print vect1.safeAccess("sick")
print vect1.norm(1)
print vect1.norm(2)
print vect1.norm(3)
