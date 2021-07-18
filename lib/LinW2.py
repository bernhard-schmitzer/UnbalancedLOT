import numpy as np
import scipy
import scipy.sparse

def extractMongeData(pi,posNu):
    """Extract approximate Monge map from optimal coupling
    
    Args:
        pi: optimal coupling in sparse.csr format
        posNu: positions of second marginal masses

    Returns:
        T: approximate transport map
    """
    
    # compute second marginal of coupling pi
    pi1=np.array(pi.sum(axis=0)).flatten()
        
    # reserve empty array for Monge map
    T=np.zeros((pi.shape[0],posNu.shape[1]),dtype=np.double)

    # go through points in barycenter
    for j in range(T.shape[0]):
        # check if current row is empty
        if pi.indptr[j+1]==pi.indptr[j]:
            continue

        # extract masses in that row of the coupling (based on csr format)
        piRow=pi.data[pi.indptr[j]:pi.indptr[j+1]]
        # normalize masses
        piRow=piRow/np.sum(piRow)
        # extract indices non-zero entries (based on csr format)
        piIndRow=pi.indices[pi.indptr[j]:pi.indptr[j+1]]
        
        # need einsum for averaging along first ("zeroth") axis
        T[j,:]=np.einsum(posNu[piIndRow],[0,1],piRow,[0],[1])
        
    return T

def buildDataMatrix(tangentList,mu):
    """Build data matrix for Euclidean PCA. Originally, a tangent vector is a velocity and the inner product is weighted with mu.
    So reweigh velocity entries with sqrt(mu) and flatten the array such that the tangent space inner product becomes the simple Euclidean
    inner product. Then PCA etc can be performed in standard form.
    """
    
    # how many pixels does center measure have, in how many spatial dimensions
    nSpace,dimSpace=tangentList[0].shape
    # how many samples
    nSamples=len(tangentList)
    
    # sqrt for measure for re-weighting
    sqrtMu=np.sqrt(mu)
    
    # allocate empty data matrix for velocity field
    dataMatrix=np.zeros((nSamples,nSpace,dimSpace),dtype=np.double)

    # write all samples into proper location in data matrix
    for i in range(nSamples):
        # write (re-scaled) velocity field in each spatial dimension
        for j in range(dimSpace):
            dataMatrix[i,:,j]=tangentList[i][:,j]
    
    # rescale with sqrt of mu
    dataMatrix=np.einsum(dataMatrix,[0,1,2],sqrtMu,[1],[0,1,2])
    # reshape into suitable shape
    dataMatrix=dataMatrix.reshape((nSamples,nSpace*dimSpace))

    
    return dataMatrix


def decomposeEigenvectors(evList,mu):
    """From the flattend and rescaled version in the data matrix recover the original tangent vector."""
    nSpace=len(mu)
    invSqrtMu=1/np.sqrt(mu)
    dimSpace=len(evList[0])//nSpace
    result=[]
    for ev in evList:
        vel=np.zeros((nSpace,dimSpace),dtype=np.double)
        for j in range(dimSpace):
            vel[:,j]=ev[j::dimSpace]
        result.append(vel)
    for i in range(len(result)):
        result[i]=np.einsum(result[i],[0,1],invSqrtMu,[0],[0,1])
    return result
