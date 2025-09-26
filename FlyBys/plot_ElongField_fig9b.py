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

def Long_Efield_comp(x,y,z):
    y = 0.2
    q = 1.0
    fac = 70.0
    d = 1.0 # dipole separation
    dp_xloc = 4.0  # center of dipole, x coord, z=y=0

    num_pos = 9

    Ex = 0
    Ey = 0
    Ez = 0
    V = 0

    x = x - dp_xloc

    Rm = (np.sqrt((x-d/2.0)**2+y**2+z**2) )

    eps_0 = 5.526349406e+5 # e^2 eV^-1 cm^-1 # 55.26349406 e2⋅eV−1⋅μm−1
    eps_w = 80.1
    eps_o = 2.25
    a = 0.5

    Gamma = (eps_o-eps_w)/(eps_o+2.0*eps_w) #-0.4792244
#    fac = q/(4.0*np.pi*eps_0*eps_w)

    for jj in range(num_pos):
        Rpterms = (np.sqrt((x+(2*jj+1)*d/2.0)**2+y**2+z**2) )**3
        Ex = Ex + (x+(2*jj+1)*d/2.0)/num_pos/Rpterms
        Ey = Ey + 1.0/num_pos/Rpterms
        Ez = Ez + 1.0/num_pos/Rpterms

        RpVs= ( np.sqrt( (x+(2*jj+1)*d/2.0)**2+y**2+z**2) )
        V = V + 1.0/num_pos/RpVs

    Ex = fac*(Ex - (x-d/2.0)/Rm**3 )
    Ey = y*fac*(Ey - 1.0/Rm**3)
    Ez = z*fac*(Ez - 1.0/Rm**3)
    V = fac*(V - 1.0/Rm)

    gamma = Gamma*a**3

    return Ex,Ey,Ez,V,gamma


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern Serif",
    "font.size":14,
})

# Meshgrid 
xx, zz = np.meshgrid(np.linspace(-6, 6, 11),
                   np.linspace(0.5, 7, 11))

xxx, zzz = np.meshgrid(np.linspace(-7, 7, 100),
                   np.linspace(0.1, 7, 100))

xo2_pts = np.arange(-7.25,0,0.5)
zo2_pts_2 = xo2_pts*0.0 + 2.0
zo2_pts_5 = xo2_pts*0.0 + 5.0

xxxx = np.linspace(-8,7,100)
print(xxxx)
zzzz2 = xxxx*0.0 + 2.0
zzzz5 = xxxx*0.0 + 5.0
yy = 0.2

# Directional vectors 
Ex,Ey,Ez,V,gamma = Long_Efield_comp(xx,yy,zz)

print(Long_Efield_comp(0.0,0.0,5.0))

mx_V = np.max(V)
mn_V = np.min(V)
print(mx_V)
print(mn_V)

pos_lev = 10**np.array([0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]) #,2.25,2.5])
print(pos_lev)

len_pos = len(pos_lev)
print(len_pos)

pot_levs=np.zeros([2*len_pos+1,])
pot_levs[0:len_pos] = -1*pos_lev[::-1]
pot_levs[len_pos] = 0.0
pot_levs[len_pos+1:] = pos_lev

print(pot_levs)

E_nrm = (np.sqrt(Ex**2+Ez**2))**(3/4)

fig1 = plt.figure(1)

plt.plot(xxxx,zzzz2,c='red')
plt.plot(xxxx,zzzz5,c='coral')
plt.plot(xo2_pts,zo2_pts_2,'.',c='red',markersize=5)
plt.plot(xo2_pts,zo2_pts_5,'.',c='coral',markersize=5)

# Plotting Vector Field with QUIVER 
Ex,Ey,Ez,V,gamma = Long_Efield_comp(xx,yy,zz)
plt.quiver(xx-2, zz, Ex/E_nrm, Ez/E_nrm)

Ex,Ey,Ez,V,gamma = Long_Efield_comp(xxx,yy,zzz)
plt.contour(xxx-2,zzz,V,pot_levs,cmap='gray')
#plt.title('Electric Field and Potential')
plt.xlabel('$x$',fontsize=20)
plt.ylabel('$z$',fontsize=20)

# Setting x, y boundary limits 
plt.xlim(-8, 1)
plt.ylim(0, 7)

#plt.savefig('Elong_field.pdf',bbox_inches='tight')
plt.show()


