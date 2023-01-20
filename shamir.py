#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:17:59 2023

@author: kst
"""

import numpy as np
import random

class Shamir:
    def __init__(self,n,t,mu,sigma,p):
        self.n = n
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.p = p

    def gen_shares(self,secret,var = 0):
        if var == 0:
            var = self.sigma
        coefficients = np.empty(self.t+1)
        shares = np.empty(self.n)
        y = np.random.normal(self.mu, var, self.t)
        
        sub = random.sample(range(self.n), self.t)
        x_es = np.empty(self.t+1)
        x_es[0] = 0
        for i in range(self.t):
            x_es[i+1] = self.p[sub[i]]
        
        for i in range(self.n):
            shares[i] = 0
            for j in range(self.t+1):
                temp = 1
                for k in range(self.t+1):
                    if j!=k:
                        temp = temp*(self.p[i]-x_es[k])/(x_es[j]-x_es[k])
                if j == 0:
                    shares[i] = shares[i]+secret*temp
                else:
                    shares[i] = shares[i]+y[j-1]*temp
                    
        return np.array(shares),y
       
    def recon_secret(self,shares,p):
        L_p = np.empty(len(p))
        for i in range(len(p)):
            L_prod = 1
            for j in range(0,len(p)):
                if p[i]!=p[j]:
                    L_prod *= (-p[j])/(p[i]-p[j])
            L_p[i] = L_prod
        sums = np.dot(shares,L_p)
        return sums
    
    def gen_triplets(self):
        a = np.random.normal(self.mu, 10**2)
        b = np.random.normal(self.mu, 10**2)
        c = a*b
        return self.gen_shares(a)[0], self.gen_shares(b)[0], self.gen_shares(c, self.sigma*1600)[0]
    
    def mult_with_triplet(self, share_1, share_2, share_a, share_b, share_c):
        
        d = share_1 - share_a
        e = share_2 - share_b
        
        d_rec = self.recon_secret(d,self.p)
        e_rec = self.recon_secret(e,self.p)
        
        res = d_rec*e_rec*np.ones(self.n) + d_rec*share_b + e_rec*share_a + share_c
        
        return res, d, e
    
    def get_inverse(self, share_1, r_shares, a, b, c, p, x_es):
        
        sr_mult = self.mult_with_triplet(share_1, r_shares, a, b, c, p, x_es)
        
        sr = self.recon_secret(sr_mult,p,x_es)
        
        s_inv = r_shares/sr
        
        return np.array(s_inv)