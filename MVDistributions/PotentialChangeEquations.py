import numpy as np

def DeltaPhi(x,r,delta_phi): 

#    eps_w = 80.1
#    eps_o = 2.25
#    a = 1.0

#    Gamma = (eps_o-eps_w)/(eps_o+2.0*eps_w) #-0.4792244
#    gamma = Gamma * a**3

#    gamma = 1.0


    Ex1 = x[0]
    Ey1 = x[1]
    Ex2 = x[2]
    Ey2 = x[3]
    Ez1 = x[4]
    Ez2 = x[5]
    x01 = x[6]
    y01 = x[7]
    z01 = x[8]
    x02 = x[9]
    y02 = x[10]
    z02 = x[11]

    N_r = r.shape[0]
    fn = np.zeros(N_r)
    for jj in range (0,N_r):
        phi1 = -((Ex1*(x01-r[jj][0])+Ey1*(y01-r[jj][1])+Ez1*z01)/
                      ((x01-r[jj][0])**2+(y01-r[jj][1])**2+z01**2)**(3/2))
        phi2 = -((Ex2*(x02-r[jj][0])+Ey2*(y02-r[jj][1])+Ez2*z02)/
                      ((x02-r[jj][0])**2+(y02-r[jj][1])**2+z02**2)**(3/2))
        fn[jj] = delta_phi[jj]-(phi2-phi1)
    return fn

