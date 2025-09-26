import numpy as np
import matplotlib.pyplot as plt
from PotentialChangeEquations import DeltaPhi
from PotentialChangeJacobian import Df
from NewtonSystemIteration import NewtSysSolve
from numpy.random import seed
from numpy.random import randn
from numpy.random import rand
from numpy.linalg import norm
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
soln[0] = gamma * 0.0 #0.2
#  Ey1 
soln[1] = gamma * 0.0 #0.1
#  Ex2 
soln[2] = gamma * 0.0 #-0.1
#  Ey2 
soln[3] = gamma * 0.0 #0.05
#  Ez1
soln[4] = gamma * 0.9
#  Ez2
soln[5] = gamma * 0.9
#  x01 
soln[6]  = -1.0
#  y01 
soln[7]  = -0.5
#  z01 
soln[8]  = 5
#  x02 
soln[9]  = 2.0
#  y02 
soln[10] = 1.5
#  z02 
soln[11] = 5

#Newton's Method setup
epsf = 1.0e-15
epsx = 1.2e-8
itmx = 20


#Receptors
r = []


xrang = [-3.5, -1.5, 1.5, 3.5]
yrang = [-3, 0, 3]

for ii in xrang:
    for jj in yrang:
        r.append([ii,jj])
r = np.array(r)        

print(r.shape)

delta_phi_ex = np.zeros(12)
delta_phi_ex = -DeltaPhi(soln,r,delta_phi_ex)

print(delta_phi_ex)

print(DeltaPhi(soln,r,delta_phi_ex))

n_trials = 400000

x_all = np.zeros([n_trials,12])

nbins = 31
binscl = 1.3/nbins #16.0 #15.0
cntbin = np.zeros([nbins,1])
convbin = np.zeros([nbins,1])
cnotbin = np.zeros([nbins,1])
cnt = 0
cnot = 0

for jj in range(x_all.shape[0]):

    scl = 1.5*rand()
    print(f'scl= {scl}')


    x0 = np.copy(soln)
    x0 = soln + scl*(rand(12)-0.5)

    diffsoln = abs(x0-soln)
    bin_indx = int(np.floor(norm(diffsoln,2)/binscl))
    if (bin_indx>nbins-1):
        bin_indx=nbins-1
    print(f'bin = {bin_indx}')
    cntbin[bin_indx]+=1



    # Newton solve
    x,err,res,its,conv,sing = NewtSysSolve(DeltaPhi, Df, x0, r, delta_phi_ex, epsx, epsf, itmx)


    #Recording
    result = 'No convergence'
    if conv==1:
        result = 'convergence'
        x_all[cnt,:] = x
        cnt = cnt + 1
        convbin[bin_indx]+=1
        for ii in range(len(x)):
            if abs(x[ii])-abs(soln[ii])>epsx:
                result = 'Incorrect convergence'
#                print(x)
                cnot = cnot + 1
                cnotbin[bin_indx]+=1
            break

print(delta_phi_ex)
print(soln)
print(f'cnt= {cnt}')
print(f'cnot= {cnot}')
x_mn = np.mean(x_all[:cnt,:],axis=0)
x_sd = np.std(x_all[:cnt,:],axis=0)
print(x_mn)
print(x_sd)
print(cntbin)
print(convbin/cntbin)
print(cnotbin/cntbin)

bins = np.arange(nbins-1)*binscl

f1,ax1 = plt.subplots()
ax1.plot(bins,convbin[:-1]/cntbin[:-1])
ax1.set_xlabel('$\| \mathbf{X}_0 - \mathbf{x}_{exact} \|_2$',fontsize=18)
ax1.set_ylabel('Convergence Fraction',fontsize=18)
#plt.axis([0,1.45,0.0,1.05])
#ax1.legend(fontsize=18)

#plt.savefig('Fig_MinArray_IntGuess_correct.pdf',bbox_inches='tight')


f2,ax2 = plt.subplots()
ax2.plot(bins,cnotbin[:-1]/cntbin[:-1])
ax2.set_xlabel('$\| \mathbf{X}_0 - \mathbf{x}_{exact} \|_2$',fontsize=18)
ax2.set_ylabel('Convergence to other Fraction',fontsize=18)
#plt.axis([0,1.45,0.0,1.05])
#ax2.legend(fontsize=18)

#plt.savefig('Fig_MinArray_IntGuess_wrong.pdf',bbox_inches='tight')

plt.show()


