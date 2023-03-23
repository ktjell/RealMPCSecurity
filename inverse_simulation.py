# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:23:10 2023

@author: Jaron
"""

import numpy as np
from AdditiveSS import AddSharing
from sklearn.covariance import EmpiricalCovariance

def sim_inverse(AS, A, H, xS, xinvA):

    rS,reshare = AS.gen_inverse()
    prodS = AS.gen_shares(AS.recon_secret(rS)*AS.recon_secret(xS))
    xA = xS[A]
    share_to_reshareH = -AS.recon_secret(reshare)-np.sum(xinvA-reshare[A])
    return np.concatenate((xA,rS[A],prodS,reshare[A],share_to_reshareH), axis=None)
   
    
def genData(n,AS,A,H,length):
    
    Views_R = np.zeros((n,length))
    Views_I = np.zeros((n,length))
    
    for i in range(n):
        #Real View
        x = (2*np.random.randint(2)-1)*np.random.normal(5, 1)
    
        xS = AS.gen_shares(x)
        rS,reshare = AS.gen_inverse()
        prodS = AS.gen_shares(AS.recon_secret(rS)*x)
        a_share,b_share,c_share = AS.gen_triplets()
        reshare_mult = AS.gen_reshare()
        
        x_inv_shares, share_to_reshare = AS.inverse(xS, rS, reshare, a_share, b_share, c_share, reshare_mult)
        
        Views_R[i,:] =  np.concatenate((xS[A], rS[A], prodS, reshare[A], share_to_reshare[H], x_inv_shares), axis = None)
        
        
        #Ideal View
        x = (2*np.random.randint(2)-1)*np.random.normal(5, 1)
    
        xS = AS.gen_shares(x)
        x_inv_shares= AS.gen_shares(1/x)
        temp = sim_inverse(AS,A, H, xS, x_inv_shares[A])
        Views_I[i,:] = np.append(temp,x_inv_shares)

        
    return Views_R,Views_I

AS = AddSharing(3,0,10,1000)
A = [1,2]
H=[0]
length = 13


View_R,View_I = genData(10000,AS,A,H,length)


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