import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import prepareLm

X1 = np.load('npy/talking_dataset.npy')
X2 = np.load('npy/silent_dataset.npy')
Y1 = np.ones(X1.shape[0])
Y2 = np.zeros(X2.shape[0])

X = np.concatenate((X1, X2), axis=0) / 150
norm = np.linalg.norm(X)

Y = np.concatenate((Y1, Y2), axis=0).reshape(-1,1)

# np.savetxt("debug/X.csv", X.transpose(2,0,1).reshape(-1,2), delimiter=",")
# np.savetxt("debug/Y.csv", Y, delimiter=",")

print(X.shape)
print(Y.shape)

n = 1

# for i in range(0,n):
#     x = X[i][:,0]
#     y = X[i][:,1]
#     plt.scatter(x, y, s = 5)
# plt.show()

with open('models/trained', 'rb') as f:
    clf = pickle.load(f)

X1 = prepareLm(X[0])

X2 = prepareLm(X[5555])

# print(X[0])
# print(X_r)

print(clf.predict(X1))
print(clf.predict(X2))

