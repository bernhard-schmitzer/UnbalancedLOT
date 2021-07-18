# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from lib.header_notebook import *
import lib.OTCommon as OTCommon
import lib.SinkhornNP as Sinkhorn
import lib.LinHK as LinHK
import lib.LinW2 as LinW2

# %matplotlib inline
# -

# # Compute HK tangent space embedding

# load reference measure, here we just take Euclidean mean
datRef=OTCommon.importMeasure("data/two_ellipses_rad0d25/mean.mat",totalMass=1.,keepZero=False)
muRef=datRef[1]
posRef=datRef[2].astype(np.double)

params={}
params["setup_HKMode"]=True
params["setup_HKScale"]=5
params["solver_errorGoal"]=1.E-4
params["aux_verbose"]=False
params["solver_epsStart"]=1E3
params["solver_epsTarget"]=0.1

# compute all tangent vectors and distances
tangentList=[]
distList=[]
for i in range(64):
    print(i)
    filename="data/two_ellipses_rad0d25/sample_{:03d}.mat".format(i)
    datSamp=OTCommon.importMeasure(filename,totalMass=1.,keepZero=False)
    muSamp=datSamp[1]
    posSamp=datSamp[2].astype(np.double)

    # solve HK transport
    value,pi=Sinkhorn.SolveHK(muRef,posRef,muSamp,posSamp,HKScale=params["setup_HKScale"],
            SinkhornError=params["solver_errorGoal"],
            epsTarget=params["solver_epsTarget"], epsInit=params["solver_epsStart"],
            returnSolver=False
            )    
    # from optimal coupling compute tangent vector data
    u0,u1,x1,refPerp,sampPerp=LinHK.extractMongeData(pi,muRef,muSamp,posSamp)
    v0,alpha0=LinHK.HKLog(posRef,u0,x1,u1,params["setup_HKScale"])
    tangentList.append((v0,alpha0,sampPerp))
    distList.append(value)

import pickle
with open("experiments/two_ellipses_rad0d25_HK5/tangent.dat", 'wb') as f:
    pickle.dump([tangentList,distList],f,2)


