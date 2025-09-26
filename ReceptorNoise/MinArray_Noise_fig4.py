import numpy as np
import matplotlib.pyplot as plt
from PotentialChangeEquations import DeltaPhi
from PotentialChangeJacobian import Df
from NewtonSystemIteration import NewtSysSolve
from numpy.random import seed
from numpy.random import randn
from numpy.linalg import norm
import warnings
import json

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "ComputerModern",
    "font.size":14,
})

soln = np.zeros(12)

eps_w = 80.1
eps_o = 2.25
a = 0.5

Gamma = (eps_o-eps_w)/(eps_o+2.0*eps_w) #-0.4792244
gamma = Gamma * a**3


#  Ex1 
soln[0] = gamma* 0.0 #0.2
#  Ey1 
soln[1] = gamma* 0.0 #0.1
#  Ex2 
soln[2] = gamma* 0.0 #-0.1
#  Ey2 
soln[3] = gamma* 0.0 #0.05
# Ez1 
soln[4] = gamma* 0.9
#  Ez2
soln[5] = gamma* 0.9
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
epsx = 1.0e-9
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

nbins = 7
ngss = nbins*10000

x_all = np.zeros([ngss,12])
e_all = np.zeros(ngss)

x_binned = np.zeros([nbins,ngss,12])

cntbin = np.zeros(nbins,dtype=int)
convbin = np.zeros(nbins,dtype=int)
cnotbin = np.zeros(nbins,dtype=int)
cnt = 0
cnot = 0
jj = 0

scl_vec = np.array([1.0e-8,3.0e-8,1.0e-7,3.0e-7,1.0e-6,3.0e-6,8.0e-6])#,3.0e-5,1.0e-4])


while (jj < x_all.shape[0]):

    scl = scl_vec[int(np.floor(jj/(ngss/nbins)))]
    bin_indx = int(np.floor(jj/(ngss/nbins)))

    cntbin[bin_indx] += 1

    print(f'scl= {scl}')

    delta_phi_nz = delta_phi_ex + scl*randn(12)

    x0 = np.copy(soln)

    diffsoln = abs(delta_phi_nz - delta_phi_ex)
    print('phi')
    print(delta_phi_nz)
    print(delta_phi_ex)
    print(f' e = {np.log10(norm(diffsoln,2)/norm(delta_phi_ex,2))}')
    e_all[jj] = np.log10(norm(diffsoln/delta_phi_ex,np.inf))
    if (bin_indx>nbins-1):
        bin_indx=nbins-1
    if (bin_indx<0):
        bin_indx=0


    print(f'bin = {bin_indx}; scale = {scl}')

    # Newton solve
    x,err,res,its,conv,sing = NewtSysSolve(DeltaPhi, Df, x0, r, delta_phi_nz, epsx, epsf, itmx)

    #Recording
    result = 'No convergence'
    if conv==1:
        result = 'convergence'
        x_all[cnt,:] = x
        x_binned[bin_indx,convbin[bin_indx],:] = x
        cnt = cnt + 1
        jj = jj + 1
        convbin[bin_indx]+=1


    print(result)

print(delta_phi_ex)
print(soln)
print(f'cnt= {cnt}')

x_mn = np.mean(x_all[:cnt,:],axis=0)
x_sd = np.std(x_all[:cnt,:],axis=0)
print(x_mn)
print(x_sd)


x_mn_b = np.zeros([nbins,12])
x_sd_b = np.zeros([nbins,12])
std_x = np.zeros(nbins)
std_y = np.zeros(nbins)
std_z = np.zeros(nbins)

min_convbin=np.min(convbin)
print(f'min bins= {min_convbin}')


for jj in range(nbins):

    x_mn_b[jj,:] = np.mean(x_binned[jj,:min_convbin,:],axis=0)

    x_sd_b[jj,:] = np.std(x_binned[jj,:min_convbin,:],axis=0)

    std_x[jj] = x_sd_b[jj,9]
    std_y[jj] = x_sd_b[jj,10]
    std_z[jj] = x_sd_b[jj,11]


#    print(jj)
#    print(soln)
    print(x_mn_b[jj,:])

var_dict = {"x_binned":x_binned.tolist(),"min_convbin":min_convbin.tolist(),
    "x_mn_b":x_mn_b.tolist(),"x_sd_b":x_sd_b.tolist(),
    "std_x":std_x.tolist(),"std_y":std_y.tolist(),"std_z":std_z.tolist(),
    "exact_soln":soln.tolist(), "cntbin":cntbin.tolist(),
    "convbin":convbin.tolist(), "eps_x":epsx,"epsf":epsf,
    "ngss":ngss,"scl_vec":scl_vec.tolist()}


print(cntbin)
print(convbin)

print(f'inf nrm randn = {norm(scl*randn(12),np.inf)}')
print(f'max dphi = {np.max(abs(delta_phi_ex))}')
print(f'min dphi = {np.min(abs(delta_phi_ex))}')
print(f'norm dphi = {norm(delta_phi_ex,2)}')

bins = np.log10(scl_vec/norm(delta_phi_ex,np.inf))

f1,ax1 = plt.subplots()
ax1.plot(bins,np.log10(std_x),label='$\sigma_x$')
ax1.plot(bins,np.log10(std_y),label='$\sigma_y$')
ax1.plot(bins,np.log10(std_z),label='$\sigma_z$')
ax1.set_xlabel('$\log_{10}( \epsilon_{rel} )$',fontsize=18)
ax1.set_ylabel('$\log_{10}(\sigma)$',fontsize=18)
#plt.axis([0,1.45,0.0,1.05])
ax1.legend(fontsize=18)

#plt.savefig('MinimalArray_noise_N10000_z5_a0p5.pdf',bbox_inches='tight')

plt.show()

