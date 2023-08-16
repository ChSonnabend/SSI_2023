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
    if len(str(l))==3: l = str(l)+"0"
    label = r"$\lambda_{class} = $ "+str(l)+", AUC = "+str(round(auc(f[:,0],f[:,1]),3)*100)+"\%" 
    if l =="1.00": label=r"$\lambda_{class} = $ 1.00, AUC = 96.4\%"
    plt.plot(f[:,0],f[:,1],label=label)

plt.grid(True)
plt.xlabel("Bkg. Mistag Rate")
plt.ylabel("Sig. Efficiency")
plt.title("Top Classification")
plt.legend(loc="best")
plt.savefig("aucs.pdf")
