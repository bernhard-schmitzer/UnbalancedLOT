import numpy as np
import scipy
import scipy.sparse

def getCostEuclidean(posX,posY,p=2,HKMode=False,HKScale=1.):
    """Compute cost matrix between point clouds posX and posY. Returns p-th power of Euclidean distance.
    If HKMode=True, the cost matrix for the soft-marginal formulation of the Hellinger--Kantorovich distance is returned.
    In this case, all distances are divided by HKScale first."""
    if posX.ndim != 2:
        raise ValueError("ndim of posX must be 2. is {:d}".format(posX.ndim))
    if posY.ndim != 2:
        raise ValueError("ndim of posY must be 2. is {:d}".format(posY.ndim))
    xres=posX.shape[0]
    yres=posY.shape[0]
    dim=posX.shape[1]
    if dim!=posY.shape[1]:
        raise ValueError("Dimensions of posX and posY are incompatible: shape[1] must agree."+\
                " Values are: {:d} and {:d}".format(dim,posY.shape[1]))

    c=np.sum((posX.reshape((xres,1,dim))-posY.reshape(1,yres,dim))**2,axis=2)
    
    if HKMode == False:
        # apply exponent other than 2
        if p!=2:
            c=c**(p/2.)
        
    else:
        # compute HK cost function
        c=np.cos(np.minimum(c**0.5/HKScale,np.pi/2))
        
        c=np.where(\
                c>=1E-15,\
                -2.*(HKScale**2)*np.log(np.maximum(1E-15,c)),
                1E10)
        
    return c

def KL(muEff,mu,muThresh=1E-15):
    """KL divergence of muEff w.r.t. mu
    muThresh: mu is assumed to be lower-bounded by muThresh,
    entries that are two small are replaced by muThresh
    this is supposed to regularize KL a bit around the singularity at mu=0
    """
    
    muReg=np.maximum(mu,muThresh)
    
    nonZero=np.where(muEff>0)
    result=np.sum(muEff[nonZero]*np.log(muEff[nonZero]/muReg[nonZero])-muEff[nonZero])
    result+=np.sum(muReg)
    return result

class TSinkhornSolverStandard:
    """This class is a simple vanilla implementation of the stabilized Sinkhorn algorithm as described in [Schmitzer: Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems, SIAM Journal on Scientific Computing (SISC) 41(3), 2019]. It implements log-stabilization via absorption, epsilon scaling and adaptive truncation of stabilized kernels. It does not implement the coarse-to-fine scheme. This will work ok on small problems with a few 100 points per marginal."""
    MSG_EXCEEDMAXITERATIONS=30101

    def __init__(self,c,muX,muY,errorGoal,rhoX=None,rhoY=None,alpha=None,beta=None,\
                eps=None,epsInit=None,epsSteps=None,epsList=None,verbose=False):
    
    
        self.verbose=verbose
    
        self.c=c
        self.muX=muX
        self.muY=muY
        self.errorGoal=errorGoal

        # the reference measure for entropic regularization is given by rhoX \otimes rhoY
        # if these are not provided, muX \otimes muY is used
        if rhoX is None:
            self.rhoX=muX
        else:
            self.rhoX=rhoX

        if rhoY is None:
            self.rhoY=muY
        else:
            self.rhoY=rhoY

        # allocate new dual variables, or use the ones provided
        if alpha is None:
            self.alpha=np.zeros(c.shape[0],dtype=np.double)
        else:
            self.alpha=alpha
        if beta is None:
            self.beta=np.zeros(c.shape[1],dtype=np.double)
        else:
            self.beta=beta
        
        # set up scaling factors, initialize with 1
        self.u=np.ones_like(self.alpha)
        self.v=np.ones_like(self.beta)
        

        # initialize kernel variables with None (no kernel computed yet)
        self.kernel=None
        self.kernelT=None
        
        # set up epsScaling
        if epsList is not None:
            self.epsList=epsList
        else:
            if eps is not None:
                if epsInit is None:
                    self.epsList=[eps]
                else:
                    if epsSteps is None:
                        # compute epsSteps such that ratio between two successive eps is bounded by 2
                        epsSteps=int((np.log(epsInit)-np.log(eps))/np.log(2)+1)
                        epsFactor=(epsInit/eps)**(1./epsSteps)
                    self.epsList=[eps*(epsInit/eps)**(1-i/epsSteps) for i in range(epsSteps+1)]
            else:
                self.epsList=None
            
        # set current value of eps to None
        self.eps=None
        
        # other parameters
        self.cfg={
            "maxIterations" : 100000,
            "maxAbsorptionLoops" : 100,
            "innerIterations" : 100,
            "absorption_scalingBound" : 1E3,
            "absorption_scalingLowerBound" : 1E3,
            "truncation_thresh" : 1E-10
        }
                

    def solve(self):
        if self.epsList is None:
            raise ValueError("epsList is None")
        for eps in self.epsList:
            self.eps=eps
            if self.verbose: print("eps: {:f}".format(self.eps))
            msg=self.solveSingle()
            if msg!=0:
                return msg
        self.generateKernel()
        return 0

    def solveSingle(self):
        nIterations=0
        nAbsorptionLoops=0


        # compute first kernel
        self.generateKernel()

        while True:
            
            # inner iterations
            self.iterate(self.cfg["innerIterations"])


            # check if need to absorb
            if self.checkAbsorb(self.cfg["absorption_scalingBound"]):
                if self.verbose: print("\tabsorbing")
                # if absorption is required
                # increase counter of consecutive absorptions
                nAbsorptionLoops+=1

                # check if too many consecutive absorptions already have happened
                if nAbsorptionLoops>self.cfg["maxAbsorptionLoops"]:
                    raise ValueError("Exceeded maximal number of absorption loops.")

                # otherwise, absorb
                self.absorb()

                self.generateKernel()


                # skip rest of this cycle and return to iterating
                continue
            # if no absorption happened, reset consecutive absorption counter
            nAbsorptionLoops=0

            # retrieve iteration accuracy error
            error=self.getError()

            if self.verbose: print("\terror: {:e}".format(error))
            if error<=self.errorGoal:
                # if numerical accuracy has been achieved, finish

                #  check if "safety absorption" is recommended
                if self.checkAbsorb(self.cfg["absorption_scalingLowerBound"]):
                    if self.verbose: print("\tsafety absorption.")
                    # if another absorption is recommended (and in particular, some iterations thereafter)
                    self.absorb()
                    self.generateKernel()
                    continue

                else:
                    # otherwisem do a final absorption
                    if self.verbose: print("\tfinal absorption")
                    self.absorb()
                    return 0

            

            # increase iteration counter
            nIterations+=self.cfg["innerIterations"]
            if nIterations>self.cfg["maxIterations"]:
                return self.MSG_EXCEEDMAXITERATIONS
            
    def generateKernel(self):

        if self.eps is None:
            raise ValueError("eps still None.")
        
        threshC=-self.eps*np.log(self.cfg["truncation_thresh"])
        cEff=self.c-self.alpha.reshape((-1,1))-self.beta.reshape((1,-1))

        indX,indY=np.nonzero(cEff<=threshC)
        val=np.exp(-cEff[indX,indY]/self.eps)*self.rhoX[indX]*self.rhoY[indY]
        self.kernel=scipy.sparse.coo_matrix((val,(indX,indY)),shape=cEff.shape)

        self.kernel=self.kernel.tocsr()
        self.kernelT=self.kernel.transpose().tocsr()

    def absorb(self):
        self.alpha+=self.eps*np.log(self.u)
        self.u[...]=1.

        self.beta+=self.eps*np.log(self.v)
        self.v[...]=1.
        
    def checkAbsorb(self,maxValue):
        if (np.max(self.u)>maxValue) or (np.max(self.v)>maxValue):
            return True
        return False

    ##############################################################
    # model specific methods, here for standard balanced OT
        
    def getError(self):
        # return L1 error of first marginal
        muXEff=self.u*(self.kernel.dot(self.v))
        if (not np.all(np.isfinite(muXEff))):
            raise ValueError("non-finite value in marginal during error computation")
        return np.sum(np.abs(muXEff-self.muX))


    def iterate(self,n):
        # standard Sinkhorn iterations
        for i in range(n):
            self.u=self.muX/(self.kernel.dot(self.v))
            self.v=self.muY/(self.kernelT.dot(self.u))
        if (not np.all(np.isfinite(self.u))):
            raise ValueError("non-finite value in scaling factor u")
        if (not np.all(np.isfinite(self.v))):
            raise ValueError("non-finite value in scaling factor v")

            
    def getScorePrimalUnreg(self):
        # primal objective without entropic term
        
        # integrate kernel (assuming that it has recently been absorbed) against cost function array
        result=0.
        xres=self.c.shape[0]
        for x in range(xres):
            i0=self.kernel.indptr[x]
            i1=self.kernel.indptr[x+1]
            result+=np.sum(self.c[x][self.kernel.indices[i0:i1]]*self.kernel.data[i0:i1])
        return result

class TSinkhornSolverKLMarginals(TSinkhornSolverStandard):
    """Adaptation of the main class to soft-marginal constraints with Kullback--Leibler divergence. Used for computation of the Hellinger--Kantorovich distance."""
    
    def __init__(self,c,muX,muY,kappa,errorGoal,rhoX=None,rhoY=None,alpha=None,beta=None,\
                eps=None,epsInit=None,epsSteps=None,epsList=None,verbose=False):
        
        TSinkhornSolverStandard.__init__(self,c,muX,muY,errorGoal,rhoX,rhoY,alpha,beta,\
                eps,epsInit,epsSteps,epsList,verbose)
        self.kappa=kappa

    def iterate(self,n):
        # unbalanced iterations for KL marginal fidelity with weight kappa, and regularization strengh eps
        
        if self.eps is None:
            raise ValueError("eps still None.")
        
        for i in range(n):
            conv=self.kernel.dot(self.v)
            convReg=np.maximum(conv,1E-100)
            self.u=np.where(conv>0,(self.muX/convReg)**(self.kappa/(self.kappa+self.eps))\
                    *np.exp(-self.alpha/(self.kappa+self.eps)),self.u)

            conv=self.kernelT.dot(self.u)
            convReg=np.maximum(conv,1E-100)
            self.v=np.where(conv>0,(self.muY/convReg)**(self.kappa/(self.kappa+self.eps))\
                    *np.exp(-self.beta/(self.kappa+self.eps)),self.v)
        if (not np.all(np.isfinite(self.u))):
            raise ValueError("non-finite value in scaling factor u")
        if (not np.all(np.isfinite(self.v))):
            raise ValueError("non-finite value in scaling factor v")

            
    def getScorePrimalUnreg(self):
        # primal objective without entropic term
        
        # integrate coupling against cost function (call base method for that)
        result=TSinkhornSolverStandard.getScorePrimalUnreg(self)

        # add KL marginal fidelity terms
        muXEff=self.u*(self.kernel.dot(self.v))
        muYEff=self.v*(self.kernelT.dot(self.u))

        result+=self.kappa*KL(muXEff,self.muX)
        result+=self.kappa*KL(muYEff,self.muY)
        
        return result
        
    def getError(self):
        # use PD gap as error
        
        # prepare current marginals and dual variables (including current scaling factors)
        muXEff=self.u*(self.kernel.dot(self.v))
        alphaEff=self.alpha+self.eps*np.log(self.u)

        muYEff=self.v*(self.kernelT.dot(self.u))
        betaEff=self.beta+self.eps*np.log(self.v)

        # evaluate PD gap
        result=0
        result+=np.sum(muXEff*alphaEff)+np.sum(muYEff*betaEff)

        nonZero=np.where(muXEff>0)
        result+=self.kappa*np.sum(muXEff[nonZero]*np.log(muXEff[nonZero]/self.muX[nonZero])-muXEff[nonZero]+np.exp(-alphaEff[nonZero]/self.kappa)*self.muX[nonZero])
        nonZero=np.where(muYEff>0)
        result+=self.kappa*np.sum(muYEff[nonZero]*np.log(muYEff[nonZero]/self.muY[nonZero])-muYEff[nonZero]+np.exp(-betaEff[nonZero]/self.kappa)*self.muY[nonZero])

        return result

def SolveW2(muX,posX,muY,posY,
        SinkhornError,
        epsTarget, epsInit,
        returnSolver=False, verbose=False
        ):
    """Computes squared W_2 distance between two measures represented by weighted point clouds
    (muX,posX) and (muY,posY).
    
    Args:
        muX: masses of first measure mass particles
        posX: positions of first measure mass particles
        muY: masses of second measure mass particles
        posY: positions of second measure mass particles
        SinkhornError: stopping criterion for unbalanced Sinkhorn, based on L1 marginal error
        epsTarget: final entropic regularization parameter
        epsInit: initial entropic regularization parameter, solver uses epsScaling with reduction factor approximately 0.5
        returnSolver: if solver object should be returned (useful for detailed subsequent analysis)
        verbose: turn on verbose mode in solver, useful for debugging
        
    Returns:
        value: approximate squared W_2 distance between two measures
        piCSR: approximate optimal coupling pi as scipy sparse CSR matrix
        SinkhornSolver: Sinkhorn solver object (if returnSolver=True)
    """

    # which cost function to use?
    c=getCostEuclidean(posX,posY)


    # create the actual solver object
    SinkhornSolver=TSinkhornSolverStandard(c,muX,muY,SinkhornError,eps=epsTarget,epsInit=epsInit,verbose=verbose)
    msg=SinkhornSolver.solve()
    if msg!=0: raise ValueError("solve: {:d}".format(msg))

    # extract unregularized primal score
    value=SinkhornSolver.getScorePrimalUnreg()
    
    piCSR=SinkhornSolver.kernel
    
    # return optimal value and coupling, (and solver, if requested)
    if returnSolver:
        return (value,piCSR,SinkhornSolver)
    else:
        return (value,piCSR)

def SolveHK(muX,posX,muY,posY,HKScale,
        SinkhornError,
        epsTarget, epsInit,
        returnSolver=False, verbose=False
        ):
    """Computes squared HK distance between two measures represented by weighted point clouds
    (muX,posX) and (muY,posY).
    All distances divided by HKScale, final result multiplied by HKScale**2.
    
    Args:
        muX: masses of first measure mass particles
        posX: positions of first measure mass particles
        muY: masses of second measure mass particles
        posY: positions of second measure mass particles
        HKScale: length scale parameter of HK
        SinkhornError: stopping criterion for unbalanced Sinkhorn, based on primal-dual gap
        epsTarget: final entropic regularization parameter
        epsInit: initial entropic regularization parameter, solver uses epsScaling with reduction factor approximately 0.5
        returnSolver: if solver object should be returned (useful for detailed subsequent analysis)
        verbose: turn on verbose mode in solver, useful for debugging
    
    Returns:
        value: approximate squared HK distance between two measures
        piCSR: approximate optimal coupling pi as scipy sparse CSR matrix
        SinkhornSolver: Sinkhorn solver object (if returnSolver=True)
    """

    # which cost function to use?
    c=getCostEuclidean(posX,posY,HKMode=True,HKScale=HKScale)

    # create the actual solver object
    SinkhornSolver=TSinkhornSolverKLMarginals(c,muX,muY,HKScale**2,SinkhornError,eps=epsTarget,epsInit=epsInit,verbose=verbose)
    msg=SinkhornSolver.solve()
    if msg!=0: raise ValueError("solve: {:d}".format(msg))
    
    # extract unregularized primal score
    value=SinkhornSolver.getScorePrimalUnreg()
    
    piCSR=SinkhornSolver.kernel

    # return optimal value and coupling, (and solver, if requested)
    if returnSolver:
        return (value,piCSR,SinkhornSolver)
    else:
        return (value,piCSR)

