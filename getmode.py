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
    p = pdf(x)
    print ('p1  ',p)
    p = np.asarray(p)
    print ('p2  ',p)
    p[p == 0] = 1e20
    print ('p3  ',p)
    #p[p > 0] = -np.log(p)
    p[p > 0] = p
    print ('p4  ',p)
    return p

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
pdf = stats.gaussian_kde(value)

bounds = []
for i in range(0, len(trace)+len(nuistrace)):
    bounds.append((S[i, 0]-1*dS[i,0], S[i,0]+1*dS[i,0]))
print ('bounds  ',bounds)
args = {'pdf': pdf}
print("Start minimization with %s = %f" % (str(S), mpdf(S, args)))
print("Start minimization with dS %s" % (str(dS)))
res = optimize.minimize(mpdf, S+dS/2, args = args, bounds = bounds, method='L-BFGS-B', options={'gtol':10e-16,'maxiter': 100, 'disp': True})
print('===================')  
print(res)
print('===================')  
print(localnuis)
print ('PDF S     ',pdf(S))
print ('PDF S-dS  ',pdf(S-dS))
print ('PDF S+dS  ',pdf(S+dS))
print ('PDF res x ',pdf(res.x))

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
