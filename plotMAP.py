import numpy as np
import json
import matplotlib.pyplot as plt
import ROOT as r
import AtlasStyle
from array import array
import argparse
import computeAc as cac
import sys



#__________________________________________________________
def doplot_nuisance(nuistrace,data_NP,plot_post, diff):

    r.gROOT.SetStyle("ATLAS")
    r.gROOT.SetBatch(1)
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
        print (nuis, data_NP['gaus_'+nuis])
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
        nx[i] = data_NP['gaus_'+nuis]
        nx_err[i] = 0
        ny_post[i] = N_nuis-i-0.15
        ny_err_post[i] = 0.
        nx_post[i] = np.mean(nuistrace[nuis])
        nx_err_post[i] = np.std(nuistrace[nuis])
        i+=1

    mg = r.TMultiGraph()
    mg2 = r.TMultiGraph()
    green = r.TGraphErrors(N_nuis,gx,gy,gx_err,gy_err)
    yellow = r.TGraphErrors(N_nuis,yx,yy,yx_err,yy_err)
    nuisances = r.TGraphErrors(N_nuis,nx,ny,nx_err,ny_err)
    yellow.SetFillColor(r.kYellow)
    yellow.SetMaximum(N_nuis+0.5)
    yellow.GetXaxis().SetLimits(-3,3)
    green.SetFillColor(r.kGreen)
    nuisances.SetMaximum(N_nuis+0.5)
    nuisances.GetXaxis().SetLimits(-3,3)
    nuisances.SetMarkerSize(1)
    nuisances.SetFillColor(0)

    nuisances_post = r.TGraphErrors(N_nuis,nx_post,ny_post,nx_err_post,ny_err_post)
    nuisances_post.SetLineColor(r.kRed)
    nuisances_post.SetFillColor(0)
    nuisances_post.SetMarkerColor(r.kRed)
    nuisances_post.SetMarkerSize(1)
    


    pad1.Draw()
    pad1.cd()
    mg.Add(yellow)
    mg.Add(green)
    mg.SetMaximum(N_nuis+0.5)
    mg.SetMinimum(0.5)
    mg.Draw("a3Y+")
    mg.GetXaxis().SetLimits(-3,3)
    mg.GetXaxis().SetLabelSize(0.03)
    mg.GetYaxis().SetLabelSize(0)
    mg.Draw("a2Y+")
    pad2.SetFrameFillStyle(0)
    pad2.SetFillStyle(4000)
    pad2.Draw()
    pad2.cd()
    mg2.Add(nuisances)
    if plot_post: mg2.Add(nuisances_post)
    mg2.SetMaximum(N_nuis+0.5)
    mg2.SetMinimum(0.5)
    mg2.Draw("apZY+")
    mg2.GetXaxis().SetLimits(-3,3)
    mg2.GetXaxis().SetLabelSize(0.03)
    mg2.GetXaxis().SetTitle("#theta/#Delta#theta")
    mg2.GetXaxis().SetTitleSize(0.03)
    mg2.GetXaxis().SetTitleOffset(1.0)
    mg2.GetXaxis().CenterTitle()
    mg2.GetYaxis().SetLabelSize(0)

    leg = r.TLegend(0.1,0.999,0.8,0.945)
    leg.SetNColumns(2)
    leg.AddEntry(nuisances,"Mode")
    leg.AddEntry(nuisances_post,"posterior")
    mg2.Draw("apZY+")
    t = r.TText()
    t.SetTextAlign(11)
    t.SetTextSize(0.012)
    t.SetTextFont(72)

    i=0
    for nuis in nuistrace:
        t.DrawText(3.2,N_nuis+1-yy[i]-0.1, nuis)
        i+=1
    t.SetTextSize(0.04)
    l = r.TLine(0,0.5,0,N_nuis+0.5)
    l.SetLineColor(r.kBlack)
    l.SetLineStyle(7)
    l.Draw()
    leg.Draw("same")
    c1.Update()
    diffname='inclusive'
    if diff!='':diffname=diff
    c1.Print('mode_MAP_np_%s.eps'%diffname)
    c1.SaveAs('mode_MAP_np_%s.png'%diffname)




#__________________________________________________________
def plot_binpost(trace,data_NP,truth,diff):
    for i in range(0, len(trace)):
        plt.hist(trace[i], 50, label='posterior', alpha=0.7)
        plt.axvline(data_NP['truth%i'%i], color = 'black', label='mode')
        plt.axvline(truth[i], color = 'red',linestyle='dashed', label='truth')
        plt.axvline(np.mean(trace[i]), color = 'green', label='mean')

#        print (data_NP['truth%i'%i]-truth[i])
        print ('post  bin%i    %f'%(i,np.mean(trace[i])))
        print ('truth bin%i    %f'%(i,truth[i]))

        plt.legend()
        plt.xlabel('truth value')
        plt.ylabel('number of samples')
        plt.savefig('truthbin%i.eps'%int(i))
        plt.close()


    ####Get AC values
    ndiff=1
    if len(trace)>4:
        ndiff=int(len(trace)/4)

    acpost=cac.computeAcList(trace,ndiff,4)
    AC_POST=[]
    AC_POST_ERR=[]

    for d in range(ndiff):
        AC_POST.append(np.mean(acpost[d]))
        AC_POST_ERR.append(np.std(acpost[d]))
    print ('AC_POST=',AC_POST)


    AC_MAP=[]
    for d in range(ndiff):
        dy_pos = 0
        dy_neg = 0
        for i in range(d*4, d*4+2):
            dy_neg+=data_NP['truth%i'%i]
        for i in range(d*4+2, d*4+4):
            dy_pos+=data_NP['truth%i'%i]
        AC_MAP.append((dy_pos-dy_neg)/(dy_pos+dy_neg))
    print ('AC_MAP=',AC_MAP)


    AC_truth=[]
    for d in range(ndiff):
        dy_pos = 0
        dy_neg = 0
        for i in range(d*4, d*4+2):
            dy_neg+=truth[i]
        for i in range(d*4+2, d*4+4):
            dy_pos+=truth[i]
        AC_truth.append((dy_pos-dy_neg)/(dy_pos+dy_neg))
    print ('AC_truth=',AC_truth)

    diffname='inclusive'
    if diff=='':
        err_truth=[0.000097]
        x = [0]
        labels = ['inclusive']
        plt.axhline(y=AC_truth)
        plt.axhspan(AC_truth[0]-err_truth[0], AC_truth[0]+err_truth[0], facecolor='0.5', alpha=0.7, label='truth')
        plt.xticks(x, labels)
        plt.plot(x, AC_MAP, 'ro', label='mode')
        plt.errorbar(x, AC_POST, xerr=[0], yerr=AC_POST_ERR, label='posterior', fmt='o')
        plt.legend()
        plt.ylabel('Ac value')
        plt.savefig('AC_comp_'+diffname+'.eps')
        plt.close()


        plt.hist(acpost[0], 50, label='posterior')
        plt.axvline(AC_MAP[0], color = 'black', label='mode')
        plt.axvline(AC_truth[0], color = 'red',linestyle='dashed', label='truth')
        plt.axvline(AC_POST[0], color = 'green',  label='mean')

        plt.legend()
        plt.xlabel('Ac value')
        plt.ylabel('number of samples')
        plt.savefig('AC_posterior_'+diffname+'.eps')
        plt.close()

    if diff=='pttt':
        diffname=diff
        err_truth=[0.000162,0.000140,0.000247]
        for b in range(len(err_truth)):
            plt.hist(acpost[b], 50, label='posterior')
            plt.axvline(AC_MAP[b], color = 'black', label='mode')
            plt.axvline(AC_truth[b], color = 'red',linestyle='dashed', label='truth')
            plt.axvline(AC_POST[b], color = 'green',  label='mean')

            plt.legend()
            plt.xlabel('Ac value')
            plt.ylabel('number of samples')
            plt.savefig('AC_posterior_'+diffname+'_bin%i.eps'%b)
            plt.close()


#__________________________________________________________
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--diff', type=str, default = "", help='differential bin')
    parser.add_argument('--mode', type=str, default = "", help='file containing the mode of the likelihood (numpy format)')
    parser.add_argument('--trace', type=str, default = "", help='file containing the trace of the truth (numpy format)')
    parser.add_argument('--traceNP', type=str, default = "", help='file containing the trace of the Nuisance Parameters (numpy format)')
    parser.add_argument('--truth', type=str, default = "", help='file containing the truth bins (json format)')
    parser.add_argument('--plotpost', action="store_true", help='Plot the posterior mean and RMS together with the mode for the pulls and constraints plot')

    args, _ = parser.parse_known_args()

    truth = None
    if args.truth!='':
        try:
            truth = json.load(open(args.truth))
        except FileNotFoundError as e:
            print ('no such file for truth ===',args.truth,'=== please check')
            sys.exit(3)

    mode = None
    if args.mode!='':
        try:
            mode = np.load(args.mode)
        except FileNotFoundError as e:
            print ('no such file for mode ===',args.mode,'=== please check')
            sys.exit(3)

    trace=None
    if args.trace!='':
        try:
            trace = np.load(args.trace)
        except FileNotFoundError as e:
            print ('no such file for trace ===',args.trace,'=== please check')
            sys.exit(3)

    traceNP=None
    if args.traceNP!='':
        try:
            traceNP = np.load(args.traceNP)
        except FileNotFoundError as e:
            print ('no such file for traceNP ===',args.traceNP,'=== please check')
            sys.exit(3)



    print (mode)    
    print (len(trace))

    if traceNP!=None:
        traceNP=traceNP.tolist()
    if mode!=None:
        mode=mode.tolist()

    if traceNP!=None and mode!=None:
        doplot_nuisance(traceNP,mode,args.plotpost, args.diff)
    else:
        print ('can not run doplot_nuisance as inputs not specified traceNP===',traceNP,'=== mode===',mode,'===')

    if truth and trace.any() and mode:        
        plot_binpost(trace,mode,truth, args.diff)
    else:
        print ('can not run plot_binpost as inputs not specified trace===',trace,'=== mode===',mode,'=== truth===',truth,'===')


