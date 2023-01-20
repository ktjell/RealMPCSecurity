#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:53:02 2023

@author: kst
"""

from shamir import Shamir
import numpy as np
import random
import pandas as pd
from scipy.interpolate import lagrange

def sim_mult(SS, p, A, H, x1A, x2A, mu,sigma):
    """
    Generates a view for the adversary (A) for the multiplication protocol. The mult protocol uses a
    triplet (a,b,c=a*b) by
    x1*x2 = d*[b] + e*[a] + [c] + de 
    where d = [x1]-[a] and e = [x2]-[b].
    The protocol should return the inputs and output of A and the shares of the triplets of A and 
    the shares of the honest parties of the values d and e, since this is what A sees in the execution.
    The shares should be consisten, that is A should get x1x2A when computing d*bA + e*aA + cA + de.
    The idea is to randomly choose a and b, and then calculate c such the the about equality holds. This
    means a*b != c, however, A cannot see this as he only receives shares of a,b and c.
    
    Input parameters
    ----------
    SS : Class of Shamirs secret sharing scheme -> with the methods generate shares and reconstruct secret.
    p : List containing the p-values (x value in shamir scheme) of both A and honest parties.
    A : List containing the indices of the A parties.
    H : List containing the indices of the honest parties.
    x1A : Shares of x1 of A.
    x2A : Shares of x2 of A.
    x1x2A : Shares of x1*x2 of A.
    mu : mean value of chosen normal variables.
    sigma : Tvariance of chosen normal variable.

    """
    #Choose a random a and b and generate the shares.
    aS,bS,cS = SS.gen_triplets()

    #Generate the shares of d and e of the adversary parties and the honest parties
    dA = x1A - aS[A]                #Make sure adversary gets his shares of d and e
    eA = x2A - bS[A]
    d = np.random.normal(mu,sigma)  #Chosse d and e randomly
    e = np.random.normal(mu,sigma)
    #Interpolate A shares of d and e to generate honest shares.
    Pd = lagrange([0] + list(p[A]), np.insert(dA,0,d))
    Pe = lagrange([0] + list(p[A]), np.insert(eA,0,e))
    dH = Pd(p[H])
    eH = Pe(p[H])
    
    return x1A, x2A, aS[A], bS[A], cS[A], dH, eH

def real_mult(SS, p, A,H, x1S, x2S):
    a,b,c = SS.gen_triplets()
    res, d,e = SS.mult_with_triplet(x1S,x2S,a,b,c)
    
    return x1S[A], x2S[A], a[A], b[A], c[A], d[H], e[H]
    
def genData(n,p,A,H,SS):
    x1 = 5
    x2 = 10
    
    mu = 0
    sigma = 10**2
    Views = []
    
    
    for i in range(n):
        x1S = SS.gen_shares(x1)[0]
        x2S = SS.gen_shares(x2)[0]
        
        bit = random.randint(0,1)
        ## Simulation
        if bit == 0:
            x1A, x2A, aA, bA, cA, dH, eH = sim_mult(SS, p, A, H, x1S[A], x2S[A], mu, sigma)
            View = [[0],
                    list(x1A),
                    list(x2A),
                    list(aA),
                    list(bA),
                    list(cA),
                    list(dH),
                    list(eH)]
            Views.append(sum(View,[]))
            
           
        ## Real view
        else:
            x1A, x2A, aA, bA, cA, dH, eH = real_mult(SS, p, A,H, x1S, x2S)
            View = [[1],
                    list(x1A),
                    list(x2A),
                    list(aA),
                    list(bA),
                    list(cA),
                    list(dH),
                    list(eH)]
            Views.append(sum(View,[]))

    y = pd.DataFrame(Views)
    columns = ['Y', 'x1A', 'x2A', 'aA', 'bA', 'cA', 'dH', 'eH']
    lA = len(A)
    lH = len(H)
    iter2 = [lA,lA,lA,lA,lA,lH,lH]
    columns2 = ['Y']
    for i,c in enumerate(columns[1:]):
        for j in range(iter2[i]):
            columns2.append(c+'_'+str(j))
    y.columns = columns2
    return y



# if __name__ == "main":
# N = 2000
# p=np.array([1,2,3,4,5])
# A=[0,1]
# H=[2,3,4]
# SS = Shamir(len(p),1,0,10**6,p)
# dataset = genData(N,p,A,H,SS)

