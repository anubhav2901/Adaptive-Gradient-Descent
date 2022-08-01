import scipy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from losses import log_loss
from activations import sigmoid


def acc(y, y_pred):
  y_pred = np.where(y_pred<=0.5, 0, 1)
  return accuracy_score(y, y_pred)

class LogisticRegression:

  def normalize(self, x):
    return (x - self.xmean) / self.xstd

  def initialize(self, x, y):
    self.m = x.shape[0]
    self.n = x.shape[1]+1

    self.xmean = np.mean(x, axis=0)
    self.xstd = np.std(x, axis=0)

    x = self.normalize(x)

    self.x = np.c_[np.ones((self.m,1)), x]
    self.y = y
    self.p = y.shape[1]
    self.w = np.random.random((self.n, self.p))
    self.w1 = self.w.copy()

    self.losses, self.losses1 = [], []
    self.accuracies, self.accuracies1 = [], []

  def gradf(self, w):
    z = np.dot(self.x, w)
    a = sigmoid(z)
    error = a - self.y
    grad = np.dot(self.x.T, error)
    return a, grad
  
  def sgdfit(self, lr=1e-2, iter=100):
    weights = self.w.copy()
    for _ in range(iter):
      a, grad = self.gradf(weights)
      weights -= lr * grad

      self.losses.append(log_loss(self.y, a))
      self.accuracies.append(acc(self.y, a.copy()))
    
    self.w = weights

  def adfit(self, lr=1e-2, iter=100):
    theta = 1e9
    lamb0 = 1e-6
    weights = self.w1.copy()

    a, grad0 = self.gradf(weights)
    weights1 = weights - lr * grad0

    self.losses1.append(log_loss(self.y, a))
    self.accuracies1.append(acc(self.y, a.copy()))

    lr0 = 1

    for _ in range(iter-1):
      a, grad1 = self.gradf(weights1)

      self.losses1.append(log_loss(self.y, a))
      self.accuracies1.append(acc(self.y, a.copy()))

      c1 = np.sqrt(1 + theta) * lr0
      norm1 = scipy.linalg.norm(weights1 - weights)
      norm2 = 2 * scipy.linalg.norm(grad1 - grad0)
      c2 = norm1 / norm2 if norm2 != 0 else 1e10
      lr1 = min(c1, c2)
      theta = lr1 / lr0
      
      weights = weights1.copy()
      weights1 -= lr1 * grad1

      lr0 = lr1
      grad0 = grad1
    
    self.w1 = weights1.copy()

  def run(self, x, y, lr=1e-2, iter=100, make_plot=True):
    self.initialize(x, y)
    self.sgdfit(lr=lr, iter=iter)
    print('Best Accuracy using SGD: ', max(self.accuracies))
    self.adfit(iter=iter)
    print('Best Accuracy using Adaptive GD: ', max(self.accuracies1))
    if make_plot:
      self.plot()

  def plot(self):
    plt.plot(self.losses, c='r')
    plt.plot(self.losses1, c='b')
    plt.title('Logistic losses')
    plt.legend(['SGD loss', 'Adaptive GD loss'])
    plt.show()

    plt.plot(self.accuracies, c='r')
    plt.plot(self.accuracies1, c='b')
    plt.title('Accuracies')
    plt.legend(['SGD accuracy', 'Adaptive GD accuracy'])
    plt.show()
