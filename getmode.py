#for likelihood mode
from scipy import stats
from scipy import optimize
import numpy as np
import json
import copy
import matplotlib.pyplot as plt


#__________________________________________________________
def mpdf(x, args):
    pdf = args['pdf']
    p = -pdf.logpdf(x)
    p = np.asarray(p)
    dp = -pdf.jac(x)
    dp = np.asarray(dp)
    return p, dp

trace=np.load('OutDir/tests/FullTrace_asimov.npy')
nuistrace=np.load('OutDir/tests/FullTrace_asimov_NP.npy')
nuistrace = nuistrace.tolist()

d = len(trace) + len(nuistrace)
r = len(trace[0])

value = np.zeros( (d, r) )
S = np.zeros( (d, 1) )
dS = np.zeros( (d, 1) )
localnuis = []

for i in range(0, len(trace)):
    value[i, :] = copy.deepcopy(trace[i])
    m = np.mean(trace[i])
    s = np.std(trace[i])
    S[i, 0] = m
    dS[i, 0] = s


i=0
for nuis in nuistrace: 
    localnuis.append(nuis)
    value[len(trace) + i, :] = copy.deepcopy(nuistrace[nuis])
    m = np.mean(nuistrace[nuis])
    s = np.std(nuistrace[nuis])
    S[len(trace) + i, 0] = m
    dS[len(trace) + i, 0] = s
    i+=1


#pdf = stats.gaussian_kde(value,bw_method='silverman')
#pdf = stats.gaussian_kde(value)

class GKDE:
  def __init__(self, value):
    self.value = copy.deepcopy(np.atleast_2d(value.astype(np.float64)))
    self.d, self.n = value.shape
    self.cov = np.cov(value, rowvar = 1, bias = False)
    self.invcov = np.linalg.inv(self.cov)
    # multiply cov by scotts factor^2 and invcov by scotts factor^-2
    self.cov *= np.power(self.n, -2./(self.d+4.0))
    self.invcov *= np.power(self.n, 2./(self.d+4.0))
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
      result[i] = np.sum(np.exp(-energy), axis = 0)
      if result[i] == 0:
        result[i] = -1e20
      else:
        result[i] = np.log(result[i])
    return result

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
      tdiff = np.matmul(self.invcov, diff)
      energy = np.sum(diff * tdiff, axis = 0)*0.5
      result[0, i] = np.sum(np.exp(-energy), axis = 0)
      for k in range(d):
        diff_energy = -0.5*tdiff[k,0] - 0.5*np.sum(diff[:, i] * self.invcov[k, :], axis = 0)
        result_diff[k, i] = np.sum(-np.exp(-energy)*diff_energy, axis = 0)
      if result[0,i] == 0:
        result[0,i] = 1e-20
    return result_diff/result
pdf = GKDE(value)


bounds = []
for i in range(0, len(trace)+len(nuistrace)):
    bounds.append((S[i, 0]-5*dS[i,0], S[i,0]+5*dS[i,0]))
print ('bounds  ',bounds)
args = {'pdf': pdf}
print("Start minimization with %s = %f, diff = %s" % (str(S), mpdf(S, args)[0], mpdf(S, args)[1]))
print("Start minimization with dS %s" % (str(dS)))
res = optimize.minimize(mpdf, S, args = args, jac = True, bounds = bounds, method='L-BFGS-B', options={'gtol':0,'maxiter': 1000, 'disp': True})
print('===================')  
print(res)
print('===================')  
print(localnuis)
print ('PDF S     ',pdf.logpdf(S))
print ('PDF S-dS  ',pdf.logpdf(S-0.5*dS))
print ('PDF S+dS  ',pdf.logpdf(S+0.5*dS))
print ('PDF res x ',pdf.logpdf(res.x))

print ('S     ',S)
print ('S-dS  ',S-dS)
print ('S+dS  ',S+dS)

####do plots
for i in range(0, len(trace)):
    plt.hist(trace[i], 50)
    plt.axvline(res.x[i], color = 'black')
    plt.savefig('truthbin%i.png'%int(i))

    print (res.x[i])
    plt.close()
i=0
for nuis in nuistrace: 
    plt.hist(nuistrace[nuis], 50)
    plt.axvline(res.x[len(trace)+i], color = 'black')
    plt.savefig('nuispar_%s.png'%nuis)
    print (res.x[len(trace)+i])
    
    plt.close()
    i+=1
