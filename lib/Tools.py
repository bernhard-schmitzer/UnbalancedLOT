import numpy as np


def rasterizePointCloudInt(mu,pos,shapeImg):
    """Similar as rasterizePointCloud but pos is assumed to be int valued. So write values of mu precisely into the right entries of the image."""
    img=np.zeros(shapeImg,dtype=np.double,order="C")

    # this accounts for multiplicities of indices correctly    
    np.add.at(img,(np.clip(pos[:,0],0,shapeImg[0]-1),np.clip(pos[:,1],0,shapeImg[1]-1)),mu)
    return img

def rasterizePointCloud(mu,pos,shapeImg):
    """Project point cloud with positions pos and weights mu to Cartesian grid of shape shapeImg. Use bi-linear interpolation for non-integer locations."""
    img=np.zeros(shapeImg,dtype=np.double,order="C")

    # now obtain weights for each corner
    posRel=pos-pos.astype(int)
    posInt=pos.astype(int)

    # top left
    weight=(1-posRel[:,0])*(1-posRel[:,1])
    posCorner=posInt
    img+=rasterizePointCloudInt(mu*weight,posCorner,shapeImg)

    # bottom left
    weight=(posRel[:,0])*(1-posRel[:,1])
    posCorner=posInt+np.array([1,0],dtype=int)
    img+=rasterizePointCloudInt(mu*weight,posCorner,shapeImg)

    # top right
    weight=(1-posRel[:,0])*(posRel[:,1])
    posCorner=posInt+np.array([0,1],dtype=int)
    img+=rasterizePointCloudInt(mu*weight,posCorner,shapeImg)

    # bottom right
    weight=(posRel[:,0])*(posRel[:,1])
    posCorner=posInt+np.array([1,1],dtype=int)
    img+=rasterizePointCloudInt(mu*weight,posCorner,shapeImg)

    return img


def PCA(dataMat,keep=None):
    """Do simple PCA on data matrix
    dataMat is assumed to be centered, matrix of shape (nSamples,dimSample).
    """
    nSamples,dim=dataMat.shape
    if dim<nSamples:
        if keep is None:
            keep=dim
        A=dataMat.transpose().dot(dataMat)/nSamples
        eigData=np.linalg.eigh(A)
        eigval=(eigData[0][-keep::])[::-1]
        eigvec=((eigData[1][:,-keep::]).transpose())[::-1]
    else:
        if keep is None:
            keep=nSamples
        A=dataMat.dot(dataMat.transpose())/nSamples
        eigData=np.linalg.eigh(A)
        eigval=(eigData[0][-keep::])[::-1]
        eigvec=((eigData[1][:,-keep::]).transpose())[::-1]

        eigvec=np.einsum(eigvec,[0,1],dataMat,[1,2],[0,2])
        # renormalize
        normList=np.linalg.norm(eigvec,axis=1)
        eigvec=np.einsum(eigvec,[0,1],1/normList,[0],[0,1])
    return eigval,eigvec

