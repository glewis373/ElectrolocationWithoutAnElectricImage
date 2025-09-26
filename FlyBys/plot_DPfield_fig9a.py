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
    V = fac*(1.0/(np.sqrt((x+d/2.0)**2+y**2+z**2) ) - 1.0/(np.sqrt((x-d/2.0)**2+y**2+z**2) ) )
    
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
yy = 0.2


xo2_pts = np.arange(-7.25,0,0.5)
zo2_pts_2 = xo2_pts*0.0 + 2.0
zo2_pts_5 = xo2_pts*0.0 + 5.0

xxxx = np.linspace(-8,7,100)
print(xxxx)
zzzz2 = xxxx*0.0 + 2.0
zzzz5 = xxxx*0.0 + 5.0
yy = 0.2


# Directional vectors 
Ex,Ey,Ez,V,gamma = Dipole_Efield_comp(xx,yy,zz)

print(Dipole_Efield_comp(0.0,0.0,5.0))
print(gamma)

mx_V = np.max(V)
mn_V = np.min(V)
print(mx_V)
print(mn_V)

pos_lev = 10**np.array([0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0])
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

plt.plot(xxxx,zzzz2,c='blue')
plt.plot(xxxx,zzzz5,c='lightseagreen')
plt.plot(xo2_pts,zo2_pts_2,'.',c='blue',markersize=5)
plt.plot(xo2_pts,zo2_pts_5,'.',c='lightseagreen',markersize=5)

# Plotting Vector Field with QUIVER 
Ex,Ey,Ez,V,gamma = Dipole_Efield_comp(xx,yy,zz)
plt.quiver(xx-2, zz, Ex/E_nrm, Ez/E_nrm)
Ex,Ey,Ez,V,gamma = Dipole_Efield_comp(xxx,yy,zzz)
plt.contour(xxx-2,zzz,V,pot_levs,cmap='gray')
#plt.title('Electric Field and Potential')
plt.xlabel('$x$',fontsize=20)
plt.ylabel('$z$',fontsize=20)

# Setting x, y boundary limits 
plt.xlim(-8, 1)
plt.ylim(0, 7)

#plt.savefig('DP_field.pdf',bbox_inches='tight')
plt.show()


