import numpy as np
import matplotlib.pyplot as plt
from PotentialChangeEquations import DeltaPhi
from PotentialChangeJacobian import Df
from numpy.random import seed
from numpy.random import randn
import numpy.linalg
import warnings

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern",
    "font.size":14,
})


soln = np.zeros(12)

eps_w = 80.1
eps_o = 2.25
a = 0.5

Gamma = (eps_o-eps_w)/(eps_o+2.0*eps_w) #-0.4792244
gamma = Gamma * a**3

print(gamma)

#  Ex1 
soln[0] = gamma * 0.0 #-0.7 # 0.0

#  Ey1 
soln[1] = gamma * 0.0 #0.5 # 0.0

#  Ex2 
soln[2] = gamma * 0.0 #0.8 # 0.0

#  Ey2 
soln[3] = gamma * 0.0 #-1.0 # 0.0

#  Ez1
soln[4] = gamma * 0.9

#  Ez2
soln[5] = gamma * 0.9

#  x01 
soln[6]  = -1.0

#  y01 
soln[7]  = -0.5

#  z01 
soln[8]  = 5.0

#  x02 
soln[9]  = 0.5

#  y02 
soln[10] = 1.0

#  z02 
soln[11] = 5.0

#Receptors
r = []

xrang = [-3.5, -1.5, 1.5, 3.5]
yrang = [-3, 0, 3]

for ii in xrang:
    for jj in yrang:
#        r.append([ii+0.05*randn(1),jj+0.05*randn(1)])
        r.append([ii,jj])
r = np.array(r) 

print(r)

delta_phi_exact = np.zeros(12)
delta_phi_exact = -DeltaPhi(soln,r,delta_phi_exact)

J = Df(soln,r,delta_phi_exact)
CondN = np.log10(np.linalg.cond(J))
rk=np.linalg.matrix_rank(J)
print(rk)


param1 = np.linspace(-3.0,3.0,401)
param2 = np.linspace(-3.0,3.0,401)
CondNum = np.zeros([param2.shape[0],param1.shape[0]])
rk_plot = np.zeros([param2.shape[0],param1.shape[0]])

for jj in range(param1.shape[0]):
    soln[9] = param1[jj]
#    soln[0] = gamma*param1[jj]

    for kk in range(param2.shape[0]):

        soln[10] = param2[kk]
#        soln[2] = gamma*param2[kk]

        #Jacobian
        J = Df(soln,r,delta_phi_exact)
        CondN = np.log10(np.linalg.cond(J))
        rk_plot[kk,jj]=np.linalg.matrix_rank(J)
        if (CondN < 30.0):
            CondNum[kk,jj] = CondN
        else:
            CondNum[kk,jj]= 30.0

plt.figure(1)
plt.contourf(param1,param2,CondNum,levels=[2,3,4,5,6,7,8,9,10,11,12])

maxCN = np.max(CondNum)
print(f'log(max(Cond Num)) = {maxCN}')


plt.colorbar()
#plt.zlabel('Condition Number')
plt.xlabel('$x^{(2)}_o$',fontsize=18)
plt.ylabel('$y^{(2)}_o$',fontsize=18)

filename=f'CondNum_Uni_vert_XvY.pdf'

#plt.savefig(filename,bbox_inches='tight')

plt.show()


