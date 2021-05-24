import numpy as np
import matplotlib.pyplot as plt

X1 = np.load('npy/landmarks_talking.npy')
X2 = np.load('npy/landmarks_silent.npy')
Y1 = np.ones(X1.shape[0])
Y2 = np.zeros(X2.shape[0])

X = np.concatenate((X1, X2), axis=0) / 150
norm = np.linalg.norm(X)

Y = np.concatenate((Y1, Y2), axis=0).reshape(-1,1)

print(X[:1])
# print(Y[:2])

x = X[0][:,0]
y = X[0][:,1]
plt.plot(x, y, 'go--', linewidth=2, markersize=12)
plt.show()