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
import time
import json

#outfile = open('temp.json','+w')

fac = 51.0
dp_dx = 6.0  # dipole separation
dp_xloc = 2.0  # center of dipole, x coord, z=y=0
obj_size = 0.5 # a

def Dipole_Efield_comp(x,y,z):

    fac = 51.0
    d = 6.0  # dipole separation
    dp_xloc = 2.0  # center of dipole, x coord, z=y=0
    
    x = x - dp_xloc
    Rp = (np.sqrt((x+d/2.0)**2+y**2+z**2) )**3
    Rm = (np.sqrt((x-d/2.0)**2+y**2+z**2) )**3

#    eps_0 = 5.526349406e+5 # e^2 eV^-1 cm^-1 # 55.26349406 e2⋅eV−1⋅μm−1
    eps_w = 80.1
    eps_o = 2.25
    a = 0.5
    
    Gamma = (eps_o-eps_w)/(eps_o+2.0*eps_w) #-0.4792244
#    fac = q/(4.0*np.pi*eps_0*eps_w)

    Ez = z*fac*(1.0/Rp - 1.0/Rm)
    Ex = fac*((x+d/2.0)/Rp - (x-d/2.0)/Rm)
    Ey = y*fac*(1.0/Rp - 1.0/Rm)
    V = fac*(1.0/Rp**(1/3) - 1.0/Rm**(1/3))
    
    gamma = Gamma*a**3
    
    return Ex,Ey,Ez,V,gamma


font1 = {'size': 18,
        }
font2 = {'size': 14,
        }

# Meshgrid 
xx, zz = np.meshgrid(np.linspace(-5, 5, 10),
                   np.linspace(2, 7, 10))
yy = 0.2

# Directional vectors 
Ex,Ey,Ez,V,gamma = Dipole_Efield_comp(xx,yy,zz)

print(Dipole_Efield_comp(0.0,0.0,5.0))


fig1 = plt.figure(10)
# Plotting Vector Field with QUIVER 
plt.quiver(xx, zz, Ex, Ez)
plt.contour(xx,zz,V,20)
plt.title('Electric Field and Potential')
plt.xlabel('x')
plt.ylabel('z')

# Setting x, y boundary limits 
plt.xlim(-7, 7)
plt.ylim(0, 7)

soln = np.zeros(12)

print(gamma)

#  Ex1 
soln[0] = 0.0 #0.2
#  Ey1 
soln[1] = 0.0 #0.1
#  Ex2 
soln[2] = 0.0 #-0.2
#  Ey2 
soln[3] = 0.0 #0.05
#  Ez1
soln[4] = 0.9
#  Ez2
soln[5] = 0.9
#  x01 
soln[6]  = -5.75 #-1.1
#  y01 
soln[7]  = 0.2 #-1.5
#  z01 
soln[8]  = 5.0
#  x02 
soln[9]  = -5.25
#  y02 
soln[10] = 0.2
#  z02 
soln[11] = 5.0


#Receptors
r = []

xrang = [-10.0, -9.5, -9.0, -8.5, -8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
yrang = [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

for ii in xrang:
    for jj in yrang:
        r.append([ii,jj])
#        r.append([ii+0.1*randn(1),jj+0.1*randn(1)])
r = np.array(r)
Nr = r.shape[0]

print(r.shape)

#  object path
xo_0 = -5.75
xo_end = 5.0
dx = 0.5
npts = int(np.floor((xo_end - xo_0)/dx))

print(npts)

ngss = 10000
scl = 1.0e-5

tol_g = 1.0e-15
tol_f = 1.0e-7
tol_x = 1.0e-8
print(f'scl={scl}, tol_g={tol_g}, tol_f={tol_f}, tol_x={tol_x}')

# initialize variables
x_binned = np.zeros([ngss,12,npts])

cntbin = np.zeros(npts,dtype=int)
convbin = np.zeros(npts,dtype=int)

x_mn_b = np.zeros([npts,12])
x_sd_b = np.zeros([npts,12])
std_x = np.zeros(npts)
std_z = np.zeros(npts)

E_bins = np.zeros(npts)
E_bins_diff = np.zeros(npts)

xnow = np.copy(soln)

yo_1 = xnow[7]
yo_2 = xnow[10]
zo_1 = xnow[8]
zo_2 = xnow[11]

    
for kk in range(npts):

#    x0 = np.copy(xnow)  # init guess is exact (no-noise) from previous step (except for first step)

    topcrono1 = time.time()
    seed(12)
 
    xo_1 = xo_0 + kk*dx
    xo_2 = xo_0 + (kk+1)*dx

    Ex1,Ey1,Ez1,V,gamma = Dipole_Efield_comp(xo_1,yo_1,zo_1)
    Ex2,Ey2,Ez2,V,gamma = Dipole_Efield_comp(xo_2,yo_2,zo_2)
    
    xnow[0] = gamma*Ex1
    xnow[1] = gamma*Ey1
    xnow[2] = gamma*Ex2
    xnow[3] = gamma*Ey2
    xnow[4] = gamma*Ez1
    xnow[5] = gamma*Ez2
    
    xnow[6] = xo_1
    xnow[9] = xo_2 

    print(gamma*Ex1,gamma*Ey1,gamma*Ez1)

    E_bins[kk]=np.sqrt(Ex1**2+Ey1**2+Ez1**2)
    E_bins_diff[kk] = np.sqrt((Ex1-Ex2)**2+(Ey1-Ey2)**2+(Ez1-Ez2)**2)

#    x0 = np.copy(xnow) # init guess is exact (no-noise)
    x0 = np.copy(xnow) + 0.01*randn(12)

    delta_phi_ex = np.zeros(Nr)
    delta_phi_ex = -DeltaPhi(xnow,r,delta_phi_ex)

   
    print(f'xo1 = {xo_1},  xo2 = {xo_2}')
    print(f'max abs delta phi: {np.max(np.abs(delta_phi_ex))}')
    print(f'mean-squared delta phi: {np.sqrt(np.mean(delta_phi_ex**2))}')

    x_all = np.zeros([ngss,12])
 
    cnt = 0
    cnot = 0
    jj = 0
    conv_flag = True

    while (jj < x_all.shape[0]):

        if (conv_flag):
            add_noise = scl*randn(Nr)
            delta_phi_nz = delta_phi_ex + add_noise
            nz_cnt = 0
        
        x0 = np.copy(xnow) + 0.01*randn(12) 

        # SciPy least squares
        result = scipy.optimize.least_squares(DeltaPhi,x0,args=[r,delta_phi_nz],jac=Df,x_scale=1.0,gtol=tol_g,ftol=tol_f,xtol=tol_x,method='lm',max_nfev=30)

        if (result.status == False):
            conv_flag = False
            nz_cnt += 1
            if nz_cnt > 30:
                conv_flag = True
                cntbin[kk]+=1

        if result.success == True:
            x_all[cnt,:] = result.x
            x_binned[convbin[kk],:,kk] = result.x
            cnt +=1
            jj +=1
            cntbin[kk]+=1
            convbin[kk]+=1
            conv_flag = True

            print(f'convergence status: {result.status},  # feval = {result.nfev}, jj = {jj}, {nz_cnt}')

    topcrono2 = time.time() 
    print(f'cnt= {cnt}')
    print(f'cntbin={cntbin[kk]}')
    print(f'elapsed time: {topcrono2-topcrono1} \n')

    x_mn = np.mean(x_all[:cnt,:],axis=0)
    x_sd = np.std(x_all[:cnt,:],axis=0,ddof=1)

    var_dict_temp= {"x_binned":x_binned.tolist(),
        "exact_soln":soln.tolist(),"xo1_0":xo_0,"xo_end":xo_end,"dx_o":dx,
        "cntbin":cntbin.tolist(),"convbin":convbin.tolist(),
        "dp_dx":dp_dx,"dp_xloc":dp_xloc, "obj_size":obj_size, "fac":fac,
        "tol_g":tol_g,"tol_f":tol_f,"tol_x":tol_x,"ngss":ngss,"scl":scl,
        "zo_1":zo_1,"yo_1":yo_1}


    with open('temp_LSD.json','w') as ftemp:    
        json.dump(var_dict_temp,ftemp)


min_convbin=np.min(convbin)
print(f'min bins= {min_convbin}')


for jj in range(npts):

    x_mn_b[jj,:] = np.mean(x_binned[:min_convbin,:,jj],axis=0)

    x_sd_b[jj,:] = np.std(x_binned[:min_convbin,:,jj],axis=0,ddof=1)
    
    std_x[jj] = abs(x_sd_b[jj,9])/abs(soln[9])
    std_z[jj] = abs(x_sd_b[jj,11])/abs(soln[11])

var_dict = {"x_binned":x_binned.tolist(),"min_convbin":min_convbin.tolist(),
        "x_mn_b":x_mn_b.tolist(),"x_sd_b":x_sd_b.tolist(),
        "std_x":std_x.tolist(),"std_z":std_z.tolist(),
        "exact_soln":soln.tolist(),"xo1_0":xo_0,"xo_end":xo_end,"dx_o":dx,
        "cntbin":cntbin.tolist(),"convbin":convbin.tolist(),
        "dp_dx":dp_dx,"dp_xloc":dp_xloc, "obj_size":obj_size, "fac":fac,
        "tol_g":tol_g,"tol_f":tol_f,"tol_x":tol_x,"ngss":ngss,"scl":scl,
        "zo_1":zo_1,"yo_1":yo_1}


#with open('DPex_d6_c2_z5_tgem15_tfem7_txem8_N10000_a0p5_n1em5.json','w') as f:
#   json.dump(var_dict,f) 

print(cntbin)
print(convbin)

bins = xo_0 + np.arange(npts)*dx

fig1 = plt.figure(4)
f1 = plt.plot(bins,std_x)
plt.ylabel('Standard Deviation in ${x}$',fontdict=font2)
plt.xlabel('$x_o^{(2)}$',fontdict=font1)
#plt.axis([0,1.45,0.0,1.05])

fig2 = plt.figure(5)
f2 = plt.plot(bins,std_z)
plt.ylabel('Standard Deviation in ${z}$',fontdict=font2)
plt.xlabel('$x_o^{(2)}$',fontdict=font1)
#plt.axis([0,1.45,0.0,1.05])

fig3 = plt.figure(6)
f3 = plt.plot(E_bins,std_x)
plt.ylabel('Standard Deviation in ${x}$',fontdict=font2)
plt.xlabel('$|E|$',fontdict=font1)
#plt.axis([0,1.45,0.0,1.05])

fig4 = plt.figure(8)
f4 = plt.plot(E_bins_diff,std_x)
plt.ylabel('Standard Deviation in ${x}$',fontdict=font2)
plt.xlabel('$|E|_{diff}$',fontdict=font1)
#plt.axis([0,1.45,0.0,1.05])


#plt.savefig('fig_noise2.pdf')
plt.show()

