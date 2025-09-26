import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from PotentialChangeEquations import DeltaPhi
from PotentialChangeJacobian import Df
from numpy.random import seed
from numpy.random import randn
from numpy.random import rand
from numpy.linalg import norm
import warnings

import json

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


#  Ex1 
soln[0] = gamma *0.0 #0.2
#  Ey1 
soln[1] = gamma *0.0 #0.1
#  Ex2 
soln[2] = gamma *-0.0 #-0.2
#  Ey2 
soln[3] = gamma *0.0 #0.05
#  Ez1
soln[4] = gamma * 0.9 
#  Ez2
soln[5] = gamma * 0.9 
#  x01 
soln[6]  = -1.0 #-1.1
#  y01 
soln[7]  = -0.5 #-1.5
#  z01 
soln[8]  = 5
#  x02 
soln[9]  = 2.0
#  y02 
soln[10] = 1.5
#  z02 
soln[11] = 5

#tolerances 
eps_f = 1.0e-7
eps_g = 1.0e-10
#eps_x = 1.0e-4
#itmx = 200


#Receptors
r = []

xrang = [-4.0,  -3.0, -2.0,  -1.0, -0.5,  0.5, 1.0,  2.0,  3.0,  4.0]
yrang = [-4.0,  -3.0, -2.0,  -1.0, -0.5,  0.5, 1.0,  2.0,  3.0,  4.0]


for ii in xrang:
    for jj in yrang:
        r.append([ii,jj])
r = np.array(r)        

Nr = r.shape[0]

print(Nr)

delta_phi_ex = np.zeros(Nr)
delta_phi_ex = -DeltaPhi(soln,r,delta_phi_ex)

nbins = 7
ngss = 10000

all_gss = ngss*nbins


x_binned = np.zeros([nbins,ngss,12])

cntbin = np.zeros(nbins,dtype=int)
convbin = np.zeros(nbins,dtype=int)
cnotbin = np.zeros(nbins,dtype=int)
cnt = 0
cnot = 0
jj = 0
conv_flag = True

scl_vec = np.array([1.0e-7,3.0e-7,1.0e-6,3.0e-6,1.0e-5,3.0e-5,9.0e-5])# ,3.0e-4,1.0e-3,3.0e-3,1.0e-2])

for bin_indx in range(nbins):

    jj = 0
    conv_flag = True
    x_current = np.zeros([ngss,12])

    while (jj < ngss):
    
        scl = scl_vec[bin_indx]
#        eps_x = scl/100.0
       eps_x = 1.0e-8

        print(f'bin={bin_indx}, scl= {scl}')

        if (conv_flag):
            add_noise = scl*randn(Nr)
            delta_phi_nz = delta_phi_ex + add_noise
            nz_cnt = 0

        if (bin_indx < 5):
            gss_pert = 1.0e-2*randn(12)
        else: 
            if (bin_indx < 6):
                gss_pert = 1.0e-1*randn(12)
                eps_f = 5.0e-7
            else:
                gss_pert = 1.0e-1*rand(12)
                eps_f = 1.0e-6

        x0 = np.copy(soln) + gss_pert

        result = scipy.optimize.least_squares(DeltaPhi,x0,args=[r,delta_phi_nz],jac=Df,gtol=eps_g,ftol=eps_f,xtol=eps_x,method='lm',max_nfev=30)

        if result.success == True:
            conv_flag = True
            x_current[convbin[bin_indx],:] = result.x
            jj +=1
            convbin[bin_indx]+=1
            cntbin[bin_indx]+=1
        if result.success == False:
            conv_flag = False
            nz_cnt += 1
            if nz_cnt > 30:
                conv_flag = True
                print(nz_cnt)
                cntbin[bin_indx]+=1

        print(f'convergence status: {result.status},  # feval = {result.nfev}, jj = {jj}, {nz_cnt}')
        print(result.x)
        print(norm(add_noise,np.inf))

    x_binned[bin_indx,:,:]=x_current


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


    print(x_mn_b[jj,:])
    print(x_sd_b[jj,:])

var_dict = {"x_binned":x_binned.tolist(),"min_convbin":min_convbin.tolist(),
    "x_mn_b":x_mn_b.tolist(),"x_sd_b":x_sd_b.tolist(),
    "std_x":std_x.tolist(),"std_y":std_y.tolist(),"std_z":std_z.tolist(),
    "exact_soln":soln.tolist(), "cntbin":cntbin.tolist(),
    "convbin":convbin.tolist(), "ngss":ngss,"scl_vec":scl_vec.tolist()}


#with open('LS10000_a0p5_z5_tf2em7_N100.json','w') as f:
#    json.dump(var_dict,f)


print(cntbin)
print(convbin)


bins = np.log10(scl_vec[0:nbins]/norm(delta_phi_ex,np.inf))

f1,ax1 = plt.subplots()
ax1.plot(bins,np.log10(std_x),label='$\sigma_x$')
ax1.plot(bins,np.log10(std_y),label='$\sigma_y$')
ax1.plot(bins,np.log10(std_z),label='$\sigma_z$')
ax1.set_xlabel('$\log_{10}( \epsilon_{rel} )$',fontsize=18)
ax1.set_ylabel('$\log_{10}(\sigma)$',fontsize=18)
plt.axis([-4.25,-1.0,-3.5,0.0])
ax1.legend(fontsize=18)

#plt.savefig('Fig10000_a0p5_N100_z5.pdf',bbox_inches='tight')

plt.show()


