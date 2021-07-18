import numpy as np
import scipy
import scipy.sparse

def extractMongeData(pi,mu,nu,posNu,zeroThresh=1E-16):
    """Extract approximate Monge map and approximate second marginal density composed
    with Monge map from optimal coupling

    Args:    
        pi: optimal coupling in sparse.csr format
        mu: masses of first marginal
        nu: masses of second marginal
        posNu: positions of second marginal masses

    Returns:
        u0: density of mu wrt first marginal of pi (see paper)
        u1: density of nu wrt second marginal of pi (already in composition with transport map T, see paper)
        T: approximate transport map (see paper)
        muPerp: singular part of mu wrt first marginal of pi (see paper)
        nuPerp: singular part of nu wrt second marginal of pi (see paper)
    """
    
    # compute marginals of coupling pi
    pi0=np.array(pi.sum(axis=1)).flatten()
    pi1=np.array(pi.sum(axis=0)).flatten()
    
    # length barycenter measure
    shapeBarycenter=mu.shape[0]

    # detect zero locations
    sptMu=(pi0>=zeroThresh)
    sptNu=(pi1>=zeroThresh)
    # complements (for computing perp measures)
    sptMuPerp=(sptMu==False)
    sptNuPerp=(sptNu==False)
    muPerp=mu*sptMuPerp
    nuPerp=nu*sptNuPerp

    # density of first marginal
    u0=np.zeros_like(mu)
    u0[sptMu]=mu[sptMu]/pi0[sptMu]
    # density of second marginal
    u1Full=np.zeros_like(nu)
    u1Full[sptNu]=nu[sptNu]/pi1[sptNu]


    # but in the end we are interested in u1Full(T(x)), i.e. the composition with the Monge map
    # this will be stored in variable u1 below and approximated by "barycentric projection"
    
    # reserve empty arrays for Monge map and second marginal density at location of Monge map
    # so u1 is supposed to become nu/pi1 evaluated at T, and for this we also use barycentric averaging
    T=np.zeros((shapeBarycenter,posNu.shape[1]),dtype=np.double)
    u1=np.zeros((shapeBarycenter,),dtype=np.double)
    
    # go through points in barycenter
    for j in range(shapeBarycenter):
        # check if current row is empty
        if pi.indptr[j+1]==pi.indptr[j]:
            continue

        # extract masses in that row of the coupling (based on csr format)
        piRow=pi.data[pi.indptr[j]:pi.indptr[j+1]]
        # normalize masses
        piRow=piRow/np.sum(piRow)
        # extract indices non-zero entries (based on csr format)
        piIndRow=pi.indices[pi.indptr[j]:pi.indptr[j+1]]
        
        # compute averages of u1 and T wrt this row of pi
        u1[j]=np.sum(u1Full[piIndRow]*piRow)
        # need einsum for averaging along first ("zeroth") axis
        T[j,:]=np.einsum(posNu[piIndRow],[0,1],piRow,[0],[1])
        
    return (u0,u1,T,muPerp,nuPerp)    

def HKLog(x0,u0,x1,u1,HKScale=1.):
    """Logarithmic map for HK metric (ignoring "teleport contribution").
    
    Args:
        x0: positions of support point masses
        u0: density of support point masses wrt first marginal of pi
        x1: Monge map
        u1: density of second marginal masses wrt second marginal of pi, composition through Monge map
        HKScale: rescaling of spatial distances
    
    Returns:
        v0: initial particle velocity field
        alpha0: initial mass growth field
    """

    # list of absolute values of transport distance: \|T(x)-x\|
    distList=np.linalg.norm(x1-x0,axis=1)/HKScale
    
    v0=np.einsum(np.sqrt(u1/(u0+1E-16))*np.sin(distList)/(distList+1E-16),[0],(x1-x0),[0,1],[0,1])
    alpha0=-2.+2.*np.sqrt(u1/(u0+1E-16))*np.cos(distList)
    
    return (v0,alpha0)

def HKExpMonge(v,alpha,HKScale=1.):
    """Exponential map for HK metric: get Monge data (ignoring "teleport contribution").
    
    Args:
        v: initial particle velocity field
        alpha: initial mass growth field
        HKScale: rescaling of spatial distances
    
    Returns:
        TField: relative transport map for each particle
        phiField: "angle" phi for each particle, between 0 and np.pi/2
        relMField: ratio m1/m0 for each particle
    """
    
    # norm of all velocity vectors
    vNorm=np.linalg.norm(v,axis=1)
    # normalized velocity vectors
    vDir=np.einsum(v,[0,1],1/(vNorm+1E-16),[0],[0,1])

    # fields a and b from exponential map proposition
    # a
    vNormScaled=vNorm/HKScale
    # b
    alphaOffset=0.5*alpha+1.
    
    # q^2 from exponential map prop
    relMField=vNormScaled**2+alphaOffset**2
    # angle
    phiField=np.arctan2(vNormScaled,alphaOffset)
    # relative transport map
    TField=np.einsum(vDir,[0,1],phiField*HKScale,[0],[0,1])
    
    return (TField,phiField,relMField)


def HKExp(mu,pos,v,alpha,HKScale=1.):
    """Exponential map for HK metric: get target measure (ignoring "teleport contribution").

    Args:
        mu: masses of initial measure
        pos: locations of initial measure
        v: initial particle velocity field
        alpha: initial mass growth field
        HKScale: rescaling of spatial distances
    
    Returns:
        expMu: masses of resulting measure
        expT: locations of resulting measure
    """
    
    TField,phiField,relMField=HKExpMonge(v,alpha,HKScale=HKScale)
    expT=pos+TField
    expMu=mu*relMField
    return (expMu,expT)


def HKInnerProduct(mu,v0,alpha0,w0,beta0,HKScale=1.):
    return (np.einsum(v0,[0,1],w0,[0,1],mu,[0],[])+HKScale**2*0.25*np.sum(alpha0*beta0*mu))


# some simple aux functions for doing PCA
def buildDataMatrix(tangentList,mu,HKScale=1.):
    """Build data matrix for Euclidean PCA. Originally, a tangent vector has a velocity and a growth field component. Inner product is weighted with mu for velocity field and with
    0.25*mu for growth field. This function flattens the tangent vectors and adds proper weight factors such that the tangent space inner product becomes the simple Euclidean
    inner product. Then PCA etc can be performed in standard form.
    """
    
    # how many pixels does center measure have, in how many spatial dimensions
    nSpace,dimSpace=tangentList[0][0].shape
    # how many samples
    nSamples=len(tangentList)
    
    # sqrt for measure for re-weighting
    sqrtMu=np.sqrt(mu)
    
    # allocate empty data matrix for velocity field and growth field
    dataMatrix=np.zeros((nSamples,nSpace,dimSpace+1),dtype=np.double)

    # write all samples into proper location in data matrix
    for i in range(nSamples):
        # write (re-scaled) velocity field in each spatial dimension
        for j in range(dimSpace):
            dataMatrix[i,:,j]=tangentList[i][0][:,j]/HKScale
        # write growth field
        # factor 0.5 corresponds to weight 1/4 for alpha in definition of Riemannian inner product
        dataMatrix[i,:,dimSpace]=tangentList[i][1]*0.5
    
    # rescale with sqrt of mu
    dataMatrix=np.einsum(dataMatrix,[0,1,2],sqrtMu,[1],[0,1,2])
    # reshape into suitable shape
    dataMatrix=dataMatrix.reshape((nSamples,nSpace*(dimSpace+1)))

    
    return dataMatrix


def decomposeEigenvectors(evList,mu,HKScale=1.):
    """From the flattend and rescaled version in the data matrix recover the original tangent vector."""
    nSpace=len(mu)
    invSqrtMu=1/np.sqrt(mu)
    dimSpace=len(evList[0])//nSpace-1
    result=[]
    for ev in evList:
        vel=np.zeros((nSpace,dimSpace),dtype=np.double)
        for j in range(dimSpace):
            vel[:,j]=ev[j::(dimSpace+1)]*HKScale
        result.append([vel,ev[dimSpace::(dimSpace+1)]])
    for i in range(len(result)):
        result[i][0]=np.einsum(result[i][0],[0,1],invSqrtMu,[0],[0,1])
        # factor 2 corresponds to inverse of 1/4 in definition of Riemannian inner product
        result[i][1]=result[i][1]*invSqrtMu*2
    return result
