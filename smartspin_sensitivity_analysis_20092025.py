# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 10:49:32 2025

@author: luciano.detommasi
"""

import numpy as np
import copy

"""
1 Volatility of the O&M cost coefficient: sigma_H = 0.25
2 Volatility of the energy saving amount coefficient: sigma_K = 0.01
3 Energy price drift effect: alpha_E = 0.0523
4 Energy price volatility effect: sigma_E = 0.0856
5 O&M trend index: delta = 1.025
6 Initial value of the O&M cost coefficient: H0 = 0.0036
7 Initial value of the energy saving amount coefficient: K0 = 0.0043
8 Initial value of the energy price: PE0 = 22.82
9 Economic lifetime of the energy efficiency system: N = 25
10 Capital cost of the energy efficiency investment: IC = 20668991
11 Annual energy cost savings guarantee: G = 3000000
12 Owners’ expected revenue share within the guarantee: alpha = 0.05
13 Owners’ excess revenue share beyond the guarantee: beta = 0.2
14 Owners’ expected rate of return: r0 = 0.031
15 Renters’ expected rate of return: rR = 0.031
16 Project interest rate: rP = 0.031
17 ESCOs’ expected rate of return: rE = 0.06
18 Owners’ expected revenue share with renters: theta = 1
19 Maximum renters’ rebound effect: phi = 0.15
20 Risk attitude of renters: rho = −20
"""

params = {
    'IC': 3840000,
    'G': 250000, 
    'theta': 0.85,
    'n': 15,
    'phi': 0.5,
    'sigma_H': 0.25, 
    'sigma_K': 0.01, 
    'alpha_E': 0.0523,
    'sigma_E': 0.0856,
    'delta': 1.025,
    'H0': 0.0036,
    'K0': 0.43,
    'PE0': 0.24,
    'N': 25,
    'alpha': 0.05,
    'beta': 0.2,
    'r0': 0.031,
    'rR': 0.031,
    'rP': 0.031,
    'rE': 0.06,
    'rho': -20,
    'eps_H': 0.01,
    'eps_P': 0.01,
    'eps_K': 0.01
    }

def evaluate_KPIs(params):
    IC = params['IC']
    sigma_H = params['sigma_H']
    sigma_K = params['sigma_K']
    alpha_E = params['alpha_E']
    sigma_E = params['sigma_E']
    delta = params['delta']
    H0 = params['H0']
    K0 = params['K0']
    PE0 = params['PE0']
    N = params['N']
    alpha = params['alpha']
    beta = params['beta']
    r0 = params['r0']
    rR = params['rR']
    rP = params['rP']
    rE = params['rE']
    rho = params['rho']
    eps_H = params['eps_H']
    eps_P = params['eps_P']
    eps_K = params['eps_K']
    phi = params['phi']
    theta = params['theta']
    n = params['n']
    G = params['G']

    t = np.arange(N+1)

    H = IC*H0*np.exp((-np.square(sigma_H)*t/2)+(sigma_H*eps_H*np.sqrt(t)))
    I_OM = (1/delta)*H

    I = np.zeros(N+1)
    I = copy.deepcopy(I_OM)
    I[0] = IC    

    PE = PE0*np.exp((alpha_E-(np.square(sigma_E)/2))*t+sigma_E*eps_P*np.sqrt(t))
    K = IC*K0*np.exp((-np.square(sigma_K)*t/2)+(sigma_K*eps_K*np.sqrt(t)))
    f = np.log(N + 1 - t)/np.log(N)

    b = phi/(1 - np.exp(-100/rho))
    a = 1 - b
    Re = a + b*np.exp(-(200*theta-100)/rho)
    Q = f*Re*K

    R_hat = Q*PE
    R_hat[0] = 0
    R = R_hat

    R_E = np.zeros(N+1)
    for i in np.arange(1, n+1):
        R_E[i] = R[i] - alpha*G - np.max([0, beta*(R[i]-G)]) 

    R_R = np.zeros(N+1)
    for i in np.arange(1, n+1):
        R_R[i] = (1-theta)*(alpha*G + np.max([0, beta*(R[i]-G)]))
    for i in np.arange(n+1, N+1):
        R_R[i] = (1-theta)*R[i]
    
    R_O = np.zeros(N+1)
    for i in np.arange(1, n+1):
        R_O[i] = theta*(alpha*G + np.max([0, beta*(R[i]-G)]))
    R_O[(n+1):(N+1)] = R[(n+1):(N+1)] - R_R[(n+1):(N+1)]  

    NPVR = np.zeros(N+1)
    NPVR[0] = 0
    for i in np.arange(1, N+1):
        NPVR[i] = R_R[i]/pow((1+rR), i)
    NPV_R = sum(NPVR)

    NPV0 = np.zeros(N+1)
    NPV0[0] = 0
    for i in np.arange(1, N+1):
        NPV0[i] = (R_O[i] - I_OM[i])/pow((1+r0), i)
    NPV_0 = sum(NPV0)

    NPVE = np.zeros(n+1)
    NPVE[0] = -IC
    for i in np.arange(1, n+1):
        NPVE[i] = (R_E[i] - I_OM[i])/pow((1+rE), i)    
    NPV_E = sum(NPVE)

    NPVP = np.zeros(N+1)
    for i in np.arange(1, N+1):
        NPVP[i] = (R[i] - I[i])/pow((1+rP), i)    
    NPV_P = sum(NPVP)

    return NPV_R, NPV_0, NPV_E, NPV_P, Re, R

def sensitivity(phi, rho):
    params['phi'] = phi
    params['rho'] = rho
    params['n'] = 15
    NPV_R, NPV_O, NPV_E, NPV_P, Re, R = evaluate_KPIs(params)
    print("Rebound factor = ", Re)
    print();
    print("n = ", params['n'])
    print("NPV renters", NPV_R)
    print("NPV building owner", NPV_O)
    print("NPV ESCO", NPV_E)

    params['n'] = 16
    NPV_R, NPV_O, NPV_E, NPV_P, Re, R = evaluate_KPIs(params)
    print();
    print("n = ", params['n'])
    print("NPV renters", NPV_R)
    print("NPV building owner", NPV_O)
    print("NPV ESCO", NPV_E)

    params['n'] = 17
    NPV_R, NPV_O, NPV_E, NPV_P, Re, R = evaluate_KPIs(params)
    print();
    print("n = ", params['n'])
    print("NPV renters", NPV_R)
    print("NPV building owner", NPV_O)
    print("NPV ESCO", NPV_E)
    print();
    print();

# stakeholders group 1
print("Stakeholders group 1")
phi = 0.1
rho = -10
print("phi = ", phi)
print("rho = ", rho)
sensitivity(0.1, -10)

# stakeholders group 2
print("Stakeholders group 2")
phi = 0.2
rho = -10
print("phi = ", phi)
print("rho = ", rho)
sensitivity(0.2, -10)

# stakeholders group 3
print("Stakeholders group 3")
phi = 0.1
rho = -20
print("phi = ", phi)
print("rho = ", rho)
sensitivity(0.1, -20)

# stakeholders group 4
print("Stakeholders group 4")
phi = 0.2
rho = -20
print("phi = ", phi)
print("rho = ", rho)
sensitivity(0.2, -20)