import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy import stats
from scipy import optimize
from PotentialChangeEquations import DeltaPhi
from PotentialChangeJacobian import Df
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


def makeGaussFx(xxx, loc, scale):
    y = np.exp(-(xxx-loc)**2/(2*scale**2)) / (scale * np.sqrt(2*np.pi))
    return y

soln = np.zeros(12)

eps_w = 80.1
eps_o = 2.25
a = 0.5

Gamma = (eps_o-eps_w)/(eps_o+2.0*eps_w) #-0.4792244
gamma = Gamma * a**3

#  scales:  1, 0.8, 0.5, 1.2  -> delta phi 2-norm ~ 0.0056
#  Ex1 
soln[0] = gamma * 1.2 / np.sqrt(3) # 0.0, #+- 1.0 / np.sqrt(3) # 1.0 / np.sqrt(2)
#  Ey1 
soln[1] = gamma * 1.2 / np.sqrt(3) # 0.0, #+- 1.0 / np.sqrt(3) # 0.0
#  Ex2 
soln[2] = gamma * 1.2 / np.sqrt(3) # 0.0, #+- 1.0 / np.sqrt(3) # 1.0 / np.sqrt(2)
#  Ey2 
soln[3] = gamma * 1.2 / np.sqrt(3) # 0.0, #+- 1.0 / np.sqrt(3) # 0.0
#  Ez1
soln[4] = gamma * 1.2 / np.sqrt(3) # 1.0, #+- 1.0 / np.sqrt(3) # 1.0 / np.sqrt(2)
#  Ez2
soln[5] = gamma * 1.2 / np.sqrt(3) # 1.0, #+- 1.0 / np.sqrt(3) # 1.0 / np.sqrt(2)

#  x01 
soln[6]  = -1.0 #-1.0 #-1.1
#  y01 
soln[7]  = 0.2 # -1.0 #-0.5 # -1.0 #-1.5
#  z01 
soln[8]  = 5
#  x02 
soln[9]  = 1.5 #2.0 #1.0
#  y02 
soln[10] = 0.2 # 1.6 #1.5 #1.0
#  z02 
soln[11] = 5

#tolerances 
eps_f = 1.0e-7
eps_g = 1.0e-10
eps_x = 1.0e-7



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

nbins = 1
ngss = nbins*10000

nz_lv = 1.0e-5

x_all = np.zeros([ngss,12])
e_all = np.zeros(ngss)

cntbin = np.zeros(nbins,dtype=int)
convbin = np.zeros(nbins,dtype=int)
cnotbin = np.zeros(nbins,dtype=int)
cnt = 0
cnot = 0
jj = 0
conv_flag = True

scl_vec = np.array([nz_lv]) 

while (jj < x_all.shape[0]):
    
    bin_indx = 0 #int(jj/(ngss/nbins))
    scl = scl_vec[bin_indx]

    delta_phi_nz = delta_phi_ex + scl*randn(Nr)

    x0 = np.copy(soln)

    diffsoln = abs(delta_phi_nz - delta_phi_ex)
    e_all[jj] = np.log10(norm(diffsoln,2)/norm(delta_phi_ex,2))

    print(f'bin={bin_indx}, scl= {scl}, norm delta phi = {norm(delta_phi_ex,2)}')

    if (conv_flag):

        add_noise = scl*randn(Nr)
        delta_phi_nz = delta_phi_ex + add_noise
        nz_cnt = 0
        gss_pert = 1.0e-2

    x0 = np.copy(soln) + gss_pert


    # SciPy least squares
    result = scipy.optimize.least_squares(DeltaPhi,x0,args=[r,delta_phi_nz],jac=Df,gtol=eps_g,ftol=eps_f,xtol=eps_x,method='lm',max_nfev=30)

    if result.success == True:
        conv_flag = True
        x_all[convbin[bin_indx],:] = result.x

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

min_convbin=np.min(convbin)
print(f'min bins= {min_convbin}')

print(cntbin)
print(convbin)

print(norm(delta_phi_ex,np.inf))

for kk in range(3):


    mx1 = x_all[:min_convbin,6+kk].max()
    mx2 = x_all[:min_convbin,9+kk].max()
    mn1 = x_all[:min_convbin,6+kk].min()
    mn2 = x_all[:min_convbin,9+kk].min()
    av1 = x_all[:min_convbin,6+kk].mean()
    av2 = x_all[:min_convbin,9+kk].mean()

    
    print(f'min/max x: {mn1},{mx1}.  min/max y: {mn2},{mx2}')

    delg = np.max(np.array([mx1-mn1,mx2-mn2]))

    print(f'delta ={delg}')

    delg = 0.625
    
    xmin1 = av1-delg/2
    xmax1 = av1+delg/2
    ymin2 = av2-delg/2
    ymax2 = av2+delg/2
    
    xedges = np.linspace(xmin1,xmax1,100)
    yedges = np.linspace(ymin2,ymax2,100)
    
    print(f'{xmin1},{xmax1},{ymin2},{ymax2}')

    H, xedges, yedges = np.histogram2d(x_all[:min_convbin,6+kk], x_all[:min_convbin,9+kk], bins=(xedges,yedges))
    H = H.T
    
    X, Y = np.meshgrid(xedges, yedges)
    
    fig7 = plt.figure(kk,figsize=(4,4))

    p1 = plt.pcolormesh(X,Y, H, cmap='rainbow')
    plt.xlim(xmin1,xmax1)
    plt.ylim(ymin2,ymax2)
    plt.gca().set_aspect('equal',adjustable='box')
    plt.grid()
    
#    plt.ylabel('density')
    if (kk==0):
        plt.xlabel('$x_o^{(1)}$',fontsize=18)
        plt.ylabel('$x_o^{(2)}$',fontsize=18)
    if (kk==1):
        plt.xlabel('$y_o^{(1)}$',fontsize=18)
        plt.ylabel('$y_o^{(2)}$',fontsize=18)
    if (kk==2):
        plt.xlabel('$z_o^{(1)}$',fontsize=18)
        plt.ylabel('$z_o^{(2)}$',fontsize=18)
    
    filename=f'MVpdf_x_scl_Uni_tiltp_{kk}.pdf'
    
#    plt.savefig(filename,bbox_inches='tight')

plt.show()

