import numpy as np
import scipy.linalg

def NewtSysSolve(f,Df,x0,r,delta_phi,epsx,epsf,itmx):

    #Setting flags
    conv = False
    sing = False

    #Setup
    err=np.ones(itmx)
    res=np.ones(itmx)
    x = x0
    its = 0
    
    #Loop over Newton-Raphson iterates
    for k in range(itmx):                                   
    
        #Residual vector
        r_vec = f(x,r,delta_phi)

        #Residual
        res[k] = scipy.linalg.norm(r_vec,2)

        #Jacobian
        J = Df(x,r,delta_phi)
        
        #print("Condition number for iteration %i: %.2e"%(its,np.linalg.cond(J)))

        #Check if matrix will be singular
        if np.linalg.det(J)==0:
            sing = True
            continue
    
        #Solve system J dx = -f
        dx = scipy.linalg.solve(J,r_vec)

        #print(scipy.linalg.norm(dx))

        #Error estimate
        err[k] = scipy.linalg.norm(dx,2)

        #Update step
        x -= dx
        
        #Test for convergence
        if res[k] < epsf and err[k] < epsx:
            conv = True 
            break

        #Update iteration
        its = k+1

    return x,err,res,its,conv,sing
