import numpy as np

def RK(A,b,k,ω=1,output_skip=1,x0=None):
    
    n,d = A.shape
    probs = np.linalg.norm(A,axis=1)**2 / np.linalg.norm(A,ord='fro')**2
    
    x = np.zeros((k,d))
    if type(x0) is np.ndarray:
        x[0] = x0
    
    xx = np.copy(x[0])
    for j in range(k-1):
        for _ in range(output_skip):
            i = np.random.choice(n,p=probs)
            xx -= ω*A[i]*(A[i]@xx-b[i])/np.linalg.norm(A[i])**2
        
        x[j+1] = xx
        
    return x

def HBM(A,b,k,α,β,x0=None):
    
    n,d = A.shape
    probs = np.ones(n)/n
    
    ATA = A.T@A
    ATb = A.T@b
    
    x = np.zeros((k,d))
    if type(x0) is np.ndarray:
        x[0] = x0
        x[1] = x0
    
    x0 = np.copy(x[0])
    x1 = np.copy(x[1])

    for j in range(k-1):
        
        x_ = np.copy(x1)
        
        grad_f = ATA@x1 - ATb #A.T@(A@x1-b)

        x1 = x1 - α*grad_f + β*(x1-x0)
        x0 = x_
        
        x[j+1] = x1
        
    return x

def minibatch_HBM(A,b,k,α,β,B=100,sampling='row_norm',x0=None):
    
    n,d = A.shape
    if sampling == 'row_norm':
        probs = np.linalg.norm(A,axis=1)**2 / np.linalg.norm(A,ord='fro')**2

    x = np.zeros((k,d))
    if type(x0) is np.ndarray:
        x[0] = x0
        x[1] = x0
        
    x0 = np.copy(x[0])
    x1 = np.copy(x[1])

    for j in range(k-1):
        
        x_ = np.copy(x1)
        
        if sampling == 'row_norm':
            i = np.random.choice(n,size=B,p=probs)
            grad_f = (1/B)*(((1/probs[i])[:,None]*A[i]).T@(A[i]@x1-b[i]))
        elif sampling == 'uniform':
            i = np.random.choice(n,size=B)
            grad_f = (n/B)*(A[i].T@(A[i]@x1-b[i]))      

        x1 = x1 - α*grad_f + β*(x1-x0)
        x0 = x_
        
        x[j+1] = x1
        
    return x

def get_params(A,c=2):
    
    n,d = A.shape
    η = np.max(np.linalg.norm(A,axis=1)**2/np.linalg.norm(A)**2)*n

    λ = np.linalg.eigvalsh(A.T@A)

    γ = λ[0]/c
    ℓ = λ[0] - γ
    L = λ[-1] + γ

    κ = λ[-1] / λ[0]
    κ_ = np.mean(λ/λ[0])

    α = 4/(np.sqrt(L)+np.sqrt(ℓ))**2
    β = (α*(L-ℓ)/4)**2 # = ((np.sqrt(κ)-1)/(np.sqrt(κ)+1))**2

    T = [[1+β-α*λ[0],-β],[1,0]]
    Λ,U = np.linalg.eig(T)
    
    κC = np.linalg.cond(U)#4/(α*np.sqrt(γ*(L-λ[0])))
    
    return λ,ℓ,L,κ,κ_,α,β,η,κC

def get_k_(ϵ,κC,β):
    return 2*np.log(ϵ/(np.sqrt(2)*κC))/np.log(β)

def getB(k_,α,β,λ,κC,mode='true'):
    
    d = len(λ)
    if mode=='true':
        return int((16*np.exp(1)*np.sum(λ)*np.log(2*d)*α**2*κC**2*k_)/(β*np.log(k_)) * max(λ[-1],np.sqrt(2*β*np.log(k_))/(α*κC*np.sqrt(k_))))
    elif mode=='approx':
        return int((16*np.exp(1)*np.sum(λ)*λ[-1]*np.log(2*d)*α**2)/(β*np.log(1/β)))
