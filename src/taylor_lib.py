import numpy as np

def series_recip(a):
    '''
    Returns the series expansion of the reciprocal of the series a. Will return
    None to indicate error in the case that a[0] == 0, in which case the series
    expansion can no longer be done as a simple taylor series. 
    
    Derived by applying Cauchy's product rule to S*S^-1 = 1, noting a_tilde[0]
    is trivial, and then using the higher order expansion of S*S^-1 in order
    to derive a recursion for all larger a_tilde[n]. 
    '''
    N=len(a)-1
    a_tilde=np.zeros(N+1)
    if a[0]==0:
        #reciprocal has singularity at 0
        return None
    a_tilde[0]=1/a[0]
    for n in range(1,N+1):
        a_tilde[n]=-sum(a[k]*a_tilde[n-k] for k in range(1,n+1))/a[0]
    return a_tilde

def series_log(a):
    '''
    Returns the series expansion of the logarithm of the series a. Will return 
    None to indicate error in the case that a[0] == 0, in which case the series
    expansion can no longer be done as a simple taylor series. 
    
    Derived by differentiating S, using Cauchy's Product rule to produce an 
    expansion for S' with S^-1, derived using prior methods, and integrating 
    term-by-term to produce an explicit expression.     
    '''
    N=len(a)-1
    a_hat=np.zeros(N+1)
    if a[0]==0:
        #logarithm has singularity at 0
        return None
    a_hat[0]=np.log(a[0])
    a_tilde=series_recip(a)
    for n in range(0,N):
        a_hat[n+1]=sum((k+1)*a[k+1]*a_tilde[n-k] for k in range(0,n+1))/(n+1)
    return a_hat
    

def series_multiply(a,b):
    '''
    Produce the product of the taylor series a and b. Ignores terms of a or b
    that are of higher order than the other has. Uses Cauchy's Product Rule. 
    '''
    N=min(len(a),len(b))-1
    c=np.zeros(N+1)
    c[0]=a[0]*b[0]
    for n in range(1,N+1):
        c[n]=sum(a[k]*b[n-k] for k in range(0,n+1))
    return c

def integrate(a, constant=0):
    '''
    Integrate the series a term by term. Can specify the constant term, by 
    default it's 0
    '''
    #we won't add in  the extra term, but technically you could. 
    N=len(a)-1
    res=np.zeros(N+1)
    for k in range(1,N+1):
        res[k]=a[k-1]/k
    res[0]=constant
    return res

def series_add(a,b):
    '''
    Just adds the series a and b term-by-term. 
    '''
    #unlike some of the previous ones, this won't work for lists as input.
    if len(a)<len(b):
        return a+b[:len(a)]
    elif len(a)>len(b):
        return a[:len(b)]+b
    else:
        return a+b
    
def fill(*args):
    '''
    Takes some by hand terms (all of the arguments except the last), and fills
    them out with zeros to get a series that only has the terms given, and has
    leading order given by the last argument
    '''
    res=np.zeros(args[-1]+1)
    for (i,a) in enumerate(args[:min(len(args)-1,args[-1]+1)]):
        res[i]=a
    return res

def scale(series, k):
    '''
    Returns the taylor expansion for f(kx) where f(x) expands like k
    '''
    return np.multiply(series, np.power(k, range(len(series))))

def series_power(series, alpha):
    '''
    computes the taylor expansion of f(x)^alpha with JCP Miller's formula
    '''
    assert(series[0] != 0)
    N = len(series) - 1
    a = series/series[0]
    b = np.zeros(N+1)
    b[0] = 1
    for k in range(1, N+1):
        b[k] = sum((alpha*(k-p)-p)*b[p]*a[k-p] for p in range(k))/k
    result = b * (series[0] ** alpha)
    return result
    
def cos(N):
    '''
    Returns the Taylor series of cosine to order N with DP
    '''
    series = np.zeros(N+1)
    cur = 1
    for n in range(0, N//2 + 1):
        series[2*n] = cur
        cur /= -(2*n+1)*(2*n+2)
    return series

def sin(N):
    '''
    Returns the Taylor series of sine to order N with DP
    '''
    series = np.zeros(N+1)
    cur = 1
    for n in range(0, (N+1)//2):
        series[2*n+1] = cur
        cur /= -(2*n+2)*(2*n+3)
    return series

def exp(N):
    '''
    Returns the taylor series of exp to order N with DP
    '''
    series = np.zeros(N+1)
    cur = 1
    for n in range(N+1):
        series[n] = cur
        cur /= (n+1)
    return series