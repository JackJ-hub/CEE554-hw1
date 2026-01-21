import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

path = Path("/Users/jack/Downloads/HW1/Bridge_Condition.txt")
data = np.loadtxt(path) #delimiter & skiprows
#print(data) 
dataset1 = data[0:20,:]
dataset2 = data[:,:]

# Q1: Scatter plot of dataset1
y1 = dataset1[:,2]
color = np.where(y1==1, 'red', 'green')
plt.figure(figsize=(8,6))
plt.scatter(dataset1[:,0], dataset1[:,1], c=color)
plt.xlabel('x1');plt.ylabel('x2');plt.title('Scatter Plot of Dataset1');plt.grid();plt.show()
