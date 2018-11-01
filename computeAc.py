def computeAc(bins,Ndiffbins,Ndybins):
    Ac=[]
    for i in range(int(Ndiffbins)):
        Npos = sum(bins[int((i+0.5)*Ndybins):int((i+1)*Ndybins)])
        Nneg = sum(bins[int(i*Ndybins):int((i+0.5)*Ndybins)])
        Ac.append(float(Npos-Nneg)/float(Npos+Nneg))
    return Ac

def computeAcList(trace,Ndiffbins,Ndybins):
    AcList = []
    for bins in zip(*trace):
        AcList.append(computeAc(bins,Ndiffbins,Ndybins))
    AcListFormated = [bins for  bins in zip(*AcList)]
    return AcListFormated

