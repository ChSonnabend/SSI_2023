import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import auc

plt.style.use('mystyle.mlstyle')

directory   = "."
txt_pattern = 'txt'

def read_file_data(file):
    lambda_weight = float(file[11:15])    
    f = np.loadtxt(file) 
    return f, lambda_weight

files = sorted([file for file in os.listdir(directory) if file.endswith(txt_pattern) and "auc" in file])
for file in files:
    f,l = read_file_data(file)
    label = r"$\lambda_{class} = $ "+str(l)+", auc = "+str(round(auc(f[:,0],f[:,1]),3)*100)+"\%" 
    if l ==1.0: label=r"$\lambda_{class} = $ "+str(l)+", auc = 96.4\%"
    plt.plot(f[:,0],f[:,1],label=label)

plt.grid(True)
plt.xlabel("Bkg. misstag rate")
plt.ylabel("Sig. efficiency")
plt.legend(loc="best")
plt.savefig("aucs.pdf")
