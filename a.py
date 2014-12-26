#!/usr/local/bin/python

import os,sys,struct,time,random
from array import array
import numpy as np
from scipy.stats import norm

#from mnist import mnist
import mnist

def get_qs(v,ref):
    ans = {}
    n = len(v)
    inds = range(n)
    inds.sort(key=lambda i: v[i])
    ix,nx = 0,float(len(ref))
    for i in inds:
        while (ix < nx) and (ref[ix] < v[i]): ix += 1
        ans[i] = ix/nx
    return np.array([ans[i] for i in range(n)])

def get_zs(v,ref):
    qs = get_qs(v,ref)
    qmin,qmax = 0.001,0.999
    qs[qs < qmin] = qmin
    qs[qs > qmax] = qmax
    return norm.ppf(qs)

def get_v(zs,ref):
    nx = len(ref)
    inds = np.array(map(int,len(ref)*norm.cdf(zs)))
    inds[inds > nx-1] = nx-1
    return ref[inds]

def get_ref():
    try:
        return get_ref.ans
    except AttributeError:
        n = 100
        ans = sum([list(get_image(i)) for i in range(n)],[])
        ans = np.array(ans)
        ans.sort()
        ans = ans[::10]
        get_ref.ans = ans
        return get_ref()

def get_image(i=None):
    ans = mnist.IMAGES[i] if i!=None else mnist.random_image()
    ans = 1.*np.array(ans)
    ans += 0.01*np.random.random(len(ans))
    return ans

def get_v_from_x(x): return get_v(x,get_ref())
def get_x_from_v(v): return get_zs(v,get_ref())

def get_x(i): return get_x_from_v(get_image(i))

def draw_w(xs,z,v,tau):
    nj,ni = xs.shape
    nk = z.shape[0]
    icov = np.dot(z,z.T)/v + np.identity(nk)/tau
    cov = np.linalg.inv(icov)
    xxx = np.dot(z,xs)/v
    #print 'cov.shape,xxx.shape = ',cov.shape,xxx.shape
    mu = np.linalg.solve(icov,xxx)
    #print 'mu = ',mu
    #print 'mu = ',mu.shape
    ans = np.zeros((nk,ni),z.dtype)
    for i in range(ni):
        ans[:,i] = np.random.multivariate_normal(mu[:,i],cov)
    return ans,mu

def draw_z(xs,w,v):
    nj,ni = xs.shape
    nk = w.shape[0]
    icov = np.dot(w,w.T)/v + np.identity(nk)
    cov = np.linalg.inv(icov)
    xxx = np.dot(w,xs.T)/v
    mu = np.linalg.solve(icov,xxx)
    ans = np.zeros((nk,nj),w.dtype)
    for j in range(nj):
        ans[:,j] = np.random.multivariate_normal(mu[:,j],cov)
    return ans

def main():
    nk = 10
    nper = 5000
    z = np.random.randn(nk,nper)
    v = 1.
    tau = 1.
    nreps = 100
    xs = np.array([get_x_from_v(get_image()) for j in range(nper)])
    for irep in range(nreps):
        #if irep%10==0:
        #    xs = np.array([get_x_from_v(get_image()) for j in range(nper)])
        print irep+1,nreps
        w,wx = draw_w(xs,z,v,tau)
        z = draw_z(xs,w,v)
        v = ((xs - np.dot(z.T,w))**2).mean()
        tau = (w**2).mean()
        print 'v,tau = ',v,tau
    return xs,wx,z


if __name__=='__main__':
    print 'hi'



