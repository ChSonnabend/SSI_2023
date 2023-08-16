import matplotlib.pyplot as plt
import os
import numpy as np

plt.style.use('mystyle.mlstyle')

directory   = "."
txt_pattern = 'txt'
max_mass    =  250

def read_file_data(file):
    lambda_weight = float(file[11:15])
    f = np.loadtxt(file) 
    return f, lambda_weight

files = sorted([file for file in os.listdir(directory) if file.endswith(txt_pattern) and "reg" in file])

for file in files:
    f,l = read_file_data(file)
    if len(str(l))==3: l = str(l)+"0"
    label = r"$\lambda_{class} = $ "+str(l) 
    plt.step(f[:,0]/max_mass,f[:,1],label=label)

plt.grid(True)
plt.xlabel(r"$ \frac{m_{predicted}-m_{true}}{m_{predicted}}$",size=20)
plt.ylabel("Prob. Density (A.U.)")
plt.ylim(0)
plt.xlim(-0.75,0.75)
plt.legend(loc="best")
plt.savefig("regs.pdf")
