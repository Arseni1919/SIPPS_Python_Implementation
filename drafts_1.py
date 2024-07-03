import numpy as np
vec=[1,2,3]
P=[0.5,0.2,0.3]
l = np.random.choice(vec,size=2,replace=False, p=P)
print(l)