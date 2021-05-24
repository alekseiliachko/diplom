import numpy as np
import tensorflow as tf
import pickle

X1 = np.load('npy/talking_dataset.npy')
X2 = np.load('npy/silent_dataset.npy')
Y1 = np.ones(X1.shape[0])
Y2 = np.zeros(X2.shape[0])

X = np.concatenate((X1, X2), axis=0) / 150
Y = np.concatenate((Y1, Y2), axis=0)

X = X.reshape(X.shape[0],-1)

print(X.shape)
print(Y.shape)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)# in this our main data is splitted into train and test

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
prediction=clf.predict(X_test)
print(metrics.accuracy_score(prediction,y_test))

with open('models/trained', 'wb') as f:
    pickle.dump(clf, f)