import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import copy

x1 = np.array([-7, -5, 1, 4, 5.])


kde = stats.gaussian_kde(x1)

class GKDE:
  def __init__(self, value):
    self.value = copy.deepcopy(np.atleast_2d(value.astype(np.float64)))
    self.d, self.n = self.value.shape
    self.cov = np.atleast_2d(np.cov(self.value, rowvar = 1, bias = False))
    self.invcov = np.linalg.inv(self.cov)
    # multiply cov by scotts factor^2 and invcov by scotts factor^-2
    # this is the default in NumPy
    self.cov *= np.power(self.n, -2./(self.d+4.0))
    self.invcov *= np.power(self.n, 2./(self.d+4.0))
    # this is the Silverman one
    #self.cov *= np.power(self.n*(self.d+2)/4.0, -2./(self.d+4.0))
    #self.invcov *= np.power(self.n*(self.d+2)/4.0, 2./(self.d+4.0))
    # to be conservative, do not scale the covariance ... assume a unit covariance in all dimensions
    self.norm = np.sqrt(np.linalg.det(2*np.pi*self.cov)) * self.n

  def logpdf(self, points):
    points = np.atleast_2d(points)
    d, m = points.shape
    if d != self.d:
      if d == 1 and m == self.d:
        points = np.reshape(points, (self.d, 1))
        d, m = points.shape
      else:
        print("Wrong dimensions")
    result = np.zeros((m,), dtype = np.float64)
    for i in range(m):
      diff = self.value - points[:, i, np.newaxis]
      tdiff = np.matmul(self.invcov, diff)
      energy = np.sum(diff * tdiff, axis = 0)*0.5
      maxE = np.amax(-energy)
      result[i] = np.sum(np.exp(-energy - maxE), axis = 0)
      result[i] = maxE + np.log(result[i])
    return result - np.log(self.norm)

  def evaluate(self, points):
    return np.exp(self.logpdf(points))

  # return array with derivative of the function w.r.t. x_i, where i = 1..d, at points
  def jac(self, points):
    points = np.atleast_2d(points)
    d, m = points.shape
    if d != self.d:
      if d == 1 and m == self.d:
        points = np.reshape(points, (self.d, 1))
        d, m = points.shape
      else:
        print("Wrong dimensions")
    result = np.zeros((1,m), dtype = np.float64)
    result_diff = np.zeros((d,m), dtype = np.float64)
    for i in range(m):
      diff = self.value - points[:, i, np.newaxis]
      #print("diff shape: ", diff.shape)

kde_danilo = GKDE(x1)

xs = np.linspace(-10, 10, num=50)
y1 = kde(xs)
kde.set_bandwidth(bw_method='silverman')
y2 = kde(xs)
kde.set_bandwidth(bw_method=kde.factor / 3.)
y3 = kde(xs)
kde.set_bandwidth(bw_method=1.0)
y4 = kde(xs)

y1_danilo = kde_danilo.evaluate(xs)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x1, np.ones(x1.shape) / (4. * x1.size), 'bo',label='Data points (rescaled)')
ax.plot(xs, y1, label='Scott (default)')
ax.plot(xs, y2, label='Silverman')
ax.plot(xs, y3, label='Const (1/3 * Silverman)')
#ax.plot(xs, y4, label='Const (1.0)')
ax.plot(xs, y1_danilo, ':', label='Scott (by Danilo)')
ax.legend()
plt.show()
