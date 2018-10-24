#for likelihood mode
from scipy import stats
from scipy import optimize
import numpy as np
import json
import copy
import matplotlib.pyplot as plt


#__________________________________________________________
#def mpdf(x, args):
#    pdf = args['pdf']
#    p = pdf(x)
    #print ('p1  ',p)
#    p = np.asarray(p)
    #print ('p2  ',p)
    #p[p == 0] = 1e20
    #print ('p3  ',p)
#    p[p > 0] = -np.log(p)
    #p[p > 0] = -p
    #print ('p4  ',p)
#    return p


#__________________________________________________________
def mpdf(x, args):
    pdf = args['pdf']
    p = -pdf.logpdf(x)
    p = np.asarray(p)
    dp = -pdf.jac(x)
    dp = np.asarray(dp)
    #print ('==============  ',p,dp)
    return p, dp

trace=np.load('OutDir/tests/FullTrace.npy')
nuistrace=np.load('OutDir/tests/FullTrace_NP.npy')
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
      maxE = np.amax(-energy)
      result[i] = np.sum(np.exp(-energy - maxE), axis = 0)
      result[i] = maxE + np.log(result[i])
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
      #print("diff shape: ", diff.shape)
      tdiff = np.matmul(self.invcov, diff)
      #print("invcov shape: ", self.invcov.shape)
      #print("tdiff shape: ", tdiff.shape)
      energy = np.sum(diff * tdiff, axis = 0)*0.5
      #print("energy shape: ", energy.shape)
      maxE = np.amax(-energy)
      #print("max energy: ", maxE)
      result[0, i] = np.sum(np.exp(-energy - maxE), axis = 0)
      for k in range(d):
        diff_energy = -0.5*tdiff[k,:] - 0.5*np.sum(diff * np.tile(self.invcov[:, k, np.newaxis], (1, self.n) ), axis = 0)
        #print("tdiff[k,:] shape", tdiff[k,:].shape)
        #print("diff shape", diff.shape)
        #print("invcov tiled shape", np.tile(self.invcov[:, k, np.newaxis], (1, self.n) ).shape)
        #print("diff_energy", k, " shape ", diff_energy.shape)
        result_diff[k, i] = np.sum(-np.exp(-energy - maxE)*diff_energy, axis = 0)
    return result_diff/result

pdf = GKDE(value)


#pdf = stats.gaussian_kde(value,bw_method='scott')
#pdf = stats.gaussian_kde(value)

bounds = []
for i in range(0, len(trace)+len(nuistrace)):
    bounds.append((S[i, 0]-5*dS[i,0], S[i,0]+5*dS[i,0]))
print ('bounds  ',bounds)
args = {'pdf': pdf}
print("Start minimization with %s = %f, diff = %s" % (str(S), mpdf(S, args)[0], mpdf(S, args)[1]))
#print("Start minimization with %s = %f" % (str(S), mpdf(S, args)))
print("Start minimization with dS %s" % (str(dS)))


#res = optimize.minimize(mpdf, S, args = args, jac = True, bounds = bounds, method='L-BFGS-B', options={'ftol':10e-20,'gtol':10e-18,'maxiter': 500, 'disp': True})
# ROOT uses gtol (it calls it EDM) 1e-3
res = optimize.minimize(mpdf, S, args = args, jac = True, bounds = bounds, method='L-BFGS-B', options={'gtol': 1e-3, 'maxiter': 500, 'disp': True})
print('===================')  
print(res)
print('===================')  
print(localnuis)
print ('PDF S     ',mpdf(S,args))
print ('PDF S-dS  ',mpdf(S-dS,args))
print ('PDF S+dS  ',mpdf(S+dS,args))
print ('PDF res x ',mpdf(res.x,args))

####do plots
for i in range(0, len(trace)):
    plt.hist(trace[i], 50)
    plt.axvline(res.x[i], color = 'black')
    plt.savefig('truthbin%i.png'%int(i))

    print (res.x[i])
    plt.close()

#i=0
#for nuis in nuistrace: 
#    plt.hist(nuistrace[nuis], 50)
#    plt.axvline(res.x[len(trace)+i], color = 'black')
#    plt.savefig('nuispar_%s.png'%nuis)
 #   print (res.x[len(trace)+i])
    
#    plt.close()
#    i+=1


import ROOT as r
#import AtlasStyle
from array import array


r.gROOT.SetStyle("ATLAS")
r.gStyle.SetPadTopMargin(0.06)
r.gStyle.SetPadBottomMargin(0.08)
r.gStyle.SetPadLeftMargin(0.05)
r.gStyle.SetPadRightMargin(0.5)
r.gROOT.ForceStyle()

c1 = r.TCanvas("c1","A Simple Graph with error bars",500,1200)
pad1 = r.TPad("pad1","",0,0,1,1)
pad2 = r.TPad("pad2","",0,0,1,1)
N_nuis = len(nuistrace)
gx = array('d', N_nuis*[0.])
gy = array('d', N_nuis*[0.])
gx_err = array('d', N_nuis*[0.])
gy_err = array('d', N_nuis*[0.])
yx = array('d', N_nuis*[0.])
yy = array('d', N_nuis*[0.])
yx_err = array('d', N_nuis*[0.])
yy_err = array('d', N_nuis*[0.])
nx = array('d', N_nuis*[0.])
ny = array('d', N_nuis*[0.])
nx_err = array('d', N_nuis*[0.])
ny_err = array('d', N_nuis*[0.])

nx_post = array('d', N_nuis*[0.])
ny_post = array('d', N_nuis*[0.])
nx_err_post = array('d', N_nuis*[0.])
ny_err_post = array('d', N_nuis*[0.])

i=0
for nuis in nuistrace:
    gx[i] = 0.
    gy[i] = i+1.
    gx_err[i] = 1.
    gy_err[i] = 1.
    yx[i] = 0.
    yy[i] = i+1.
    yx_err[i] = 2.
    yy_err[i] = 1.
    ny[i] = N_nuis-i
    ny_err[i] = 0.
    nx[i] = res.x[len(trace)+i]
    nx_err[i] = 0
    ny_post[i] = N_nuis-i-0.15
    ny_err_post[i] = 0.

    nx_post[i] = np.mean(nuistrace[nuis])
    nx_err_post[i] = np.std(nuistrace[nuis])

    i+=1

mg = r.TMultiGraph();
mg2 = r.TMultiGraph();
green = r.TGraphErrors(N_nuis,gx,gy,gx_err,gy_err);
yellow = r.TGraphErrors(N_nuis,yx,yy,yx_err,yy_err);
nuisances = r.TGraphErrors(N_nuis,nx,ny,nx_err,ny_err);
yellow.SetFillColor(r.kYellow);
yellow.SetMaximum(N_nuis+0.5);
yellow.GetXaxis().SetLimits(-3,3);
green.SetFillColor(r.kGreen);
nuisances.SetMaximum(N_nuis+0.5);
nuisances.GetXaxis().SetLimits(-3,3);
nuisances.SetMarkerSize(1);
nuisances.SetFillColor(0);

nuisances_post = r.TGraphErrors(N_nuis,nx_post,ny_post,nx_err_post,ny_err_post);
nuisances_post.SetLineColor(r.kRed);
nuisances_post.SetFillColor(0);
nuisances_post.SetMarkerColor(r.kRed);
nuisances_post.SetMarkerSize(1);



pad1.Draw();
pad1.cd();
mg.Add(yellow);
mg.Add(green);
mg.SetMaximum(N_nuis+0.5);
mg.SetMinimum(0.5);
mg.Draw("a3Y+");
mg.GetXaxis().SetLimits(-3,3);
mg.GetXaxis().SetLabelSize(0.03);
mg.GetYaxis().SetLabelSize(0);
mg.Draw("a2Y+");
pad2.SetFrameFillStyle(0);
pad2.SetFillStyle(4000);
pad2.Draw();
pad2.cd();
mg2.Add(nuisances);
mg2.Add(nuisances_post);
mg2.SetMaximum(N_nuis+0.5);
mg2.SetMinimum(0.5);
mg2.Draw("apZY+");
mg2.GetXaxis().SetLimits(-3,3);
mg2.GetXaxis().SetLabelSize(0.03);
mg2.GetXaxis().SetTitle("#theta/#Delta#theta");
mg2.GetXaxis().SetTitleSize(0.03);
mg2.GetXaxis().SetTitleOffset(1.0);
mg2.GetXaxis().CenterTitle();
mg2.GetYaxis().SetLabelSize(0);

leg = r.TLegend(0.1,0.999,0.8,0.945);
leg.SetNColumns(2);
leg.AddEntry(nuisances,"Mode");
leg.AddEntry(nuisances_post,"posterior");
mg2.Draw("apZY+");
t = r.TText();
t.SetTextAlign(11);
t.SetTextSize(0.012);
t.SetTextFont(72);
i=0
for nuis in nuistrace:
    t.DrawText(3.2,N_nuis+1-yy[i]-0.1, nuis)
    i+=1
t.SetTextSize(0.04);
l = r.TLine(0,0.5,0,N_nuis+0.5);
l.SetLineColor(r.kBlack);
l.SetLineStyle(7);
l.Draw();
leg.Draw("same");
c1.Update();
c1.Print('mode_np.eps');
c1.SaveAs('mode_np.png');


