# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:23:10 2023

@author: Jaron
"""

import numpy as np
from AdditiveSS import AddSharing
from sklearn.covariance import EmpiricalCovariance

def sim_mult(AS, A, H, x1A, x2A, x1x2A):
    
    a,b,c = AS.gen_triplets()
    # dS = AS.gen_shares(0)-a
    # eS = AS.gen_shares(0)-b
    dA = x1A-a[A]
    eA = x2A-b[A]
    dH = -np.sum(dA)+np.sum(a)
    eH = -np.sum(eA)+np.sum(b)
    d = np.sum(dA)+np.sum(dH)
    e = np.sum(eA)+np.sum(eH)
    r = np.random.normal(AS.mu,AS.sigma_r)
    x1x2H = np.random.normal(AS.mu,AS.sigma_s)
    x1x2 = np.sum(x1x2H)+np.sum(x1x2A)
    x1x2S = np.append(x1x2H,x1x2A)
    rS = x1x2S -(x1x2-r)*np.append(np.zeros(AS.n-1),1)
    
    xy_rH = d*b[H]+e*a[H]+c[H]-rS[H]
    
    
    return np.concatenate((x1A,x2A,a[A],b[A],c[A],rS[A],dH,eH,xy_rH), axis=None)
   
    
def genData(n,AS,A,H,length):
    
    Views_R = np.zeros((n,length))
    Views_I = np.zeros((n,length))
    
    for i in range(n):
        x1 = np.random.normal(2, 1)
        x2 = np.random.normal(1, 1)
        
        a,b,c = AS.gen_triplets()
        r = AS.gen_reshare()
        x1S = AS.gen_shares(x1)
        x2S = AS.gen_shares(x2)
        res,d,e,lin = AS.mult_with_triplet(x1S,x2S,a,b,c,r)
        Views_R[i,:] =  np.concatenate((x1S[A], x2S[A], a[A], b[A], c[A], r[A], d[H], e[H], lin[H], res), axis = None)
        
        
        x1 = np.random.normal(2, 1)
        x2 = np.random.normal(1, 1)
        
        a,b,c = AS.gen_triplets()
        r = AS.gen_reshare()
        x1S = AS.gen_shares(x1)
        x2S = AS.gen_shares(x2)
        res,d,e,lin = AS.mult_with_triplet(x1S,x2S,a,b,c,r)
        temp = sim_mult(AS,A, H, x1S[A], x2S[A], res[A])
        Views_I[i,:] = np.append(temp,res)

        
    return Views_R,Views_I

AS = AddSharing(3,0,1000,10)
A = [1,2]
H=[0]
length = 18


View_R,View_I = genData(1000000,AS,A,H,length)


cov_R = EmpiricalCovariance().fit(View_R)
cov_R = cov_R.covariance_
cov_I = EmpiricalCovariance().fit(View_I)
cov_I = cov_I.covariance_
mean_I = np.mean(View_I,axis=0)
mean_R = np.mean(View_R,axis=0)
max_I = np.max(View_I,axis=0)
max_R = np.max(View_R,axis=0)
min_I = np.min(View_I,axis=0)
min_R = np.min(View_R,axis=0)
# cov_e = EmpiricalCovariance().fit(e_prime.reshape(-1,1))
# cov_e = cov_e.covariance_
# cov_d = EmpiricalCovariance().fit(d_prime.reshape(-1,1))
# cov_d = cov_d.covariance_
# mat = np.dot(np.linalg.inv(cov_I),cov_R)-np.eye(12)
# np.linalg.eigvals(mat)
# eig = np.linalg.eigvals(mat)
# 1/2*(np.trace(mat)-np.log(np.linalg.det(np.dot(cov_R,np.linalg.inv(cov_I)))))**(1/2)