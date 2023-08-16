import matplotlib.pyplot as plt
import os
import numpy as np

plt.style.use('mystyle.mlstyle')

directory   = "."
txt_pattern = '*.txt'

def read_file_data(file):
    lambda_weight = float(file[10:14])    
    f = np.readtxt(file,) 
    return f, lambda_weight

files = [file for file in os.listdir(directory) if file.endswith(txt_pattern)]
for file in files:
    f,l = read_file_data(file)
    plt.plot(f[:,0],[:,1],label=r"$\lambda_{class}=$"lambda_weight)

plt.grid(True)
plt.xlabel("Bkg. mistag rate")
plt.ylabel("Sig. efficiency")
plt.legend(loc="best")
plt.savefig()
