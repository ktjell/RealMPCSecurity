# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:51:56 2023

@author: Jaron
"""
import numpy as np

class AddSharing:
    def __init__(self,n,mu,sigma_r,sigma_s):
        self.n = n
        self.mu = mu
        self.sigma_r = sigma_r
        self.sigma_r_p = sigma_r
        self.sigma_s = sigma_s
        self.power = 1.1

    def gen_shares(self,secret):
        y = np.random.normal(self.mu, self.sigma_s, self.n-1)
        return np.append(y,secret-np.sum(y))
       
    def recon_secret(self,shares):
        return np.sum(shares)
    
    def gen_reshare(self):
        r = np.random.normal(self.mu,self.sigma_r)
        return self.gen_shares(r)
        
    def gen_triplets(self):
        a = np.random.normal(self.mu, self.sigma_r)
        b = np.random.normal(self.mu, self.sigma_r)
        c = a*b
        return self.gen_shares(a), self.gen_shares(b), self.gen_shares(c)
    
    def mult_with_triplet(self, share_1, share_2, share_a, share_b, share_c, share_r):
        d = share_1 - share_a
        e = share_2 - share_b
        
        d_rec = self.recon_secret(d)
        e_rec = self.recon_secret(e)
        
        lin = d_rec*e_rec*np.append(np.zeros(self.n-1),1) + d_rec*share_b + e_rec*share_a + share_c - share_r
        
        lin_open = self.recon_secret(lin)
        res = lin_open*np.append(np.zeros(self.n-1),1)+share_r
        
        return res, d, e, lin
    
    def gen_power_a(self,a):
        r_prime = np.random.normal(3*self.sigma_r_p, self.sigma_r_p)
        b = 2*np.random.randint(2)-1
        r = (self.power**r_prime)*b
        rma = r**(-a)
        return self.gen_shares(r), self.gen_shares(rma)
    
    def power_a(self, share, a, share_r, share_rma, share_a, share_b, share_c):
        xr_shares = self.mult_with_triplet(share, share_r, share_a, share_b, share_c)
        xr = self.recon_secret(xr_shares,self.p)
        return (xr**a)*share_rma, xr_shares
    
    def gen_inverse(self):
        r_prime = np.random.normal(3*self.sigma_r_p, self.sigma_r_p)
        b = 2*np.random.randint(2)-1
        r = (self.power**r_prime)*b
        reshare = self.gen_reshare()
        return self.gen_shares(r), reshare
    
    def inverse(self, share_x, share_r, reshare, a_share, b_share, c_share, reshare_mult):
        prod_share = self.mult_with_triplet(share_x, share_r, a_share, b_share, c_share, reshare_mult)[0]
        prod = self.recon_secret(prod_share)
        share_to_reshare = 1/prod * share_r - reshare
        return self.recon_secret(share_to_reshare)*np.append(np.zeros(self.n-1),1)+reshare, share_to_reshare
        
    
    