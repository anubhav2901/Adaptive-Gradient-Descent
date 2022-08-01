import numpy as np
from logisticregression import LogisticRegression
from sklearn.datasets import make_classification

#run test with different number of features [4, 10, 20]


#test 1
X, y = make_classification(n_features=4)
y = y.reshape((-1, 1))

logres = LogisticRegression()

logres.run(X, y)

#test 2
X, y = make_classification(n_features=10)
y = y.reshape((-1, 1))

logres = LogisticRegression()

logres.run(X, y)

#test 3
X, y = make_classification(n_features=20)
y = y.reshape((-1, 1))

logres = LogisticRegression()

logres.run(X, y)
