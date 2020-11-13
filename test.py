import numpy as np


beta = np.deg2rad(2)
R1 = R3 = np.eye(3)
R2 = np.array([[np.cos(beta),0,-np.sin(beta)],[0,1,0],[np.sin(beta),0,np.cos(beta)]])



Rtot = R1@R2@R3
print(R2)
print(Rtot)