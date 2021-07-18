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

# # Compute Wasserstein-2 tangent space embedding

# load reference measure, here we just take Euclidean mean
datRef=OTCommon.importMeasure("data/two_ellipses_rad0d25/mean.mat",totalMass=1.,keepZero=False)
muRef=datRef[1]
posRef=datRef[2].astype(np.double)

params={}
params["setup_HKMode"]=False
params["solver_errorGoal"]=1.E-3
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
    
    # solve W transport
    value,pi=Sinkhorn.SolveW2(muRef,posRef,muSamp,posSamp,
            SinkhornError=params["solver_errorGoal"],
            epsTarget=params["solver_epsTarget"], epsInit=params["solver_epsStart"],
            returnSolver=False
            )
    # extract approximate Monge map (which is logarithmic map, up to subtracting initial locations)
    T=LinW2.extractMongeData(pi,posSamp)
    tangentList.append(T-posRef)
    distList.append(value)

import pickle
with open("experiments/two_ellipses_rad0d25_W/tangent.dat", 'wb') as f:
    pickle.dump([tangentList,distList],f,2)
