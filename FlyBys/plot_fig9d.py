import numpy as np
import matplotlib.pyplot as plt

import json

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern",
    "font.size":14,
})

font1 = {'size': 18,
        }
font2 = {'size': 14,
        }


with open('DP_ExData_z5_10000_fig9d.json','r') as f:
    data_dict = json.load(f)

x_binned_dp = np.array(data_dict["x_binned"])
min_convbin_dp = np.array(data_dict["min_convbin"])

xo1_0 = data_dict["xo1_0"]
xo_end = data_dict["xo_end"]
dx_o= data_dict["dx_o"]

exact_soln_z = 5.0

npts = 15 
bins = xo1_0 + (np.arange(npts)+1)*dx_o - 2
print(npts)
print(bins)


with open('Elong_ExData_z5_10000_fig9d.json','r') as f:
    data_dict = json.load(f)

x_binned_el = np.array(data_dict["x_binned"])
min_convbin_el = np.array(data_dict["min_convbin"])


x_sd_dp = np.zeros([npts,12])
x_sd_el = np.zeros([npts,12])

for jj in range(npts):

    x_sd_dp[jj,:] = np.std(x_binned_dp[:min_convbin_dp,:,jj],axis=0,ddof=1)
    x_sd_el[jj,:] = np.std(x_binned_el[:min_convbin_el,:,jj],axis=0,ddof=1)

std_x_dp = x_sd_dp[:,9]
std_z_dp = x_sd_dp[:,11]
std_x_el = x_sd_el[:,9]
std_z_el = x_sd_el[:,11]


f1,ax1 = plt.subplots()

ax1.plot(bins,std_x_dp,label='$\sigma_x$ dipole', c='lightseagreen', linewidth=1.5)
ax1.plot(bins,std_z_dp,label='$\sigma_z$ dipole', c='lightseagreen', linestyle='dashed', linewidth=1.5)
ax1.plot(bins,std_x_el,label='$\sigma_x$ elong', c='coral', linewidth=1.5)
ax1.plot(bins,std_z_el,label='$\sigma_z$ elong', c='coral', linestyle='dashed',linewidth=1.5)


ax1.set_xlabel('$x_o^{(2)}$',fontsize=22)
plt.xlim(-8,1)
ax1.legend(fontsize=18)

#plt.savefig('DPvEL_a0p5_nem5_z5_10000.pdf',bbox_inches='tight')

plt.show()


