import numpy as np
import scipy
import scipy.io as sciio

def importMeasure(fn,totalMass=None,keepZero=False):
    """Import measure from mat file.
    
    Args:
        fn: filename
        totalMass: if not None, measure will be normalized to this total mass
        keepZero: if True, points with zero mass will be kept, otherwise removed from list of points
    
    Returns:
        dat: image of measure as dense array
        mu: list of (nonzero) masses
        pos: list of coordinates (with nonzero masses)
        shape: shape of measure array
    """
        
    dat=np.array(sciio.loadmat(fn)["a"],dtype=np.double,order="C")
    mu,pos=processDensity_Grid(dat,totalMass=totalMass,keepZero=keepZero)
    return (dat,mu,pos,dat.shape)

def getPoslistNCube(shape,dtype=np.double):
    """Create list of positions in an n-dimensional cube of size shape."""
    ndim=len(shape)

    axGrids=[np.arange(i,dtype=dtype) for i in shape]
    prePos=np.array(np.meshgrid(*axGrids,indexing='ij'),dtype=dtype )
    # the first dimension of prepos is the dimension of the  posvector, the successive dimensions are in the cube
    # so need to move first axis to end, and then flatten
    pos=np.rollaxis(prePos,0,ndim+1)
    # flattening
    newshape=(-1,ndim)
    return (pos.reshape(newshape)).copy()

def processDensity_Grid(x,totalMass=None,constOffset=None,keepZero=True,zeroThresh=1E-14):

    # process actual density
    
    # copy, cast to double and reshape
    img=np.array(x,dtype=np.double,order="C").copy()
    shape=img.shape
    nPoints=np.prod(shape)
    dim=len(shape)
    img=img.reshape((nPoints))
    
    processDensity(img,totalMass=totalMass,constOffset=constOffset)

    # get grid pos information
    posList=getPoslistNCube(shape,dtype=np.int)
    posList=posList.reshape((nPoints,dim))

    # if desired, throw away points with zero mass
    if not keepZero:
        nonZeroPos=np.nonzero(img>zeroThresh)
        img=img[nonZeroPos]
        posList=posList[nonZeroPos]
        
        # if necessary, rescale mass once more
        processDensity(img, totalMass=totalMass, constOffset=None)

    return (img,posList)

def processDensity(x, totalMass=None, constOffset=None):
    # re-normalize and add offset if required
    if totalMass is not None:
        x[:]=totalMass*x/np.sum(x)
        if constOffset is not None:
            x[:]=x+constOffset
            x[:]=totalMass*x/np.sum(x)
    else:
        if constOffset is not None:
            x[:]=x+constOffset
            #x[:]=x/np.sum(x)

