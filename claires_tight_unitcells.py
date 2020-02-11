import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sys
from decimal import Decimal
import math
import yaml

open_param_file = open('parameter_files/parameters.yaml')
param = yaml.load(open_param_file)
c = param['constants']['c']
eps_b = np.sqrt(param['n_b'])
hbar_eVs = param['constants']['hbar_eVs']
e = param['constants']['e']
prec = param['precision']

inputs = np.loadtxt(param['inputs'],skiprows=1)*1E-7

rhomb_centers = inputs[:,0:2]
dip_centers = inputs[:,2:4]
D_l_vecs = inputs[:,4:6]
D_s_vecs = inputs[:,6:8]
Q_vecs = inputs[:,8:10]
numRhombs = len(inputs)/4

D_l_wsp = 1.30/hbar_eVs
D_s_wsp = 1.62/hbar_eVs
Q_wsp = 1.80/hbar_eVs

m_DL = 2.48E-34
m_DS = 3.02E-34
m_Q = 1.25E-33

gam_NR_long = 0.45 /hbar_eVs
gam_NR_short = 0.74 / hbar_eVs
gam_NR_quad = 0.16 / hbar_eVs

# rhomb_i = 1
# rhomb_j = 0
# DL_i = np.column_stack(( dip_centers[4*(rhomb_i) : 4*(rhomb_i)+4, :], D_l_vecs[4*(rhomb_i) : 4*(rhomb_i)+4, :] ))
# DL_j = np.column_stack(( dip_centers[4*(rhomb_j) : 4*(rhomb_j)+4, :], D_l_vecs[4*(rhomb_j) : 4*(rhomb_j)+4, :] ))

def make_g(mode_n, mode_m,m,k): #mode 1,2 are four columns: [sph_cent_x, sph_cent_y, vec_x, vec_y] and four rows corresponding to the four spheres
    gtot=0
    for sph_n in range(0,4):
        for sph_m in range(0,4):
            r_nm = mode_n[sph_n,0:2]-mode_m[sph_m,0:2] #distance between the nth and mth sphere
            mag_rnm = np.linalg.norm(r_nm)
            nhat_nm = r_nm / mag_rnm
            xn_hat = mode_n[sph_n, 2:4]/np.linalg.norm(mode_n[sph_n, 2:4])
            xm_hat = mode_m[sph_m, 2:4]/np.linalg.norm(mode_m[sph_m, 2:4])
            xn_dot_nn_dot_xm = np.dot(xn_hat, nhat_nm)*np.dot(nhat_nm, xm_hat)
            nearField = ( 3.*xn_dot_nn_dot_xm - np.dot(xn_hat,xm_hat) ) / mag_rnm**3
            intermedField = 1j*k*(3*xn_dot_nn_dot_xm - np.dot(xn_hat,xn_hat)) / mag_rnm**2 
            farField = k**2*(xn_dot_nn_dot_xm - np.dot(xn_hat,xm_hat)) / mag_rnm
            g_ind = e**2 * ( nearField - intermedField - farField ) * np.exp(1j*k*mag_rnm)  
            gtot=g_ind+gtot
    return -gtot/m
#make_g(mode_n=DL_i, mode_m=DL_j,m=m_DL,k=1)


def make_H(k):
    H = np.zeros( (np.int(3*numRhombs),np.int(3*numRhombs)),dtype=complex) 
    eigval_thisround = k*c/np.sqrt(eps_b) #1/s

    gam_long = gam_NR_long + (eigval_thisround)**2*(2.0*e**2)/(3.0*m_DL*c**3)
    gam_short = gam_NR_short + (eigval_thisround)**2*(2.0*e**2)/(3.0*m_DS*c**3)
    gam_quad = gam_NR_quad + (eigval_thisround)**2*(2.0*e**2)/(3.0*m_Q*c**3)

    for i in range(0, numRhombs): #handle the on diagonal terms 
        H[3*i,3*i]     = ( D_l_wsp**2 - 1j*eigval_thisround*gam_long ) / eigval_thisround
        H[3*i+1,3*i+1] = ( D_s_wsp**2 - 1j*eigval_thisround*gam_short  ) / eigval_thisround
        H[3*i+2,3*i+2] = ( Q_wsp**2 - 1j*eigval_thisround*gam_quad  ) / eigval_thisround
    for rhomb_i in range(0 , numRhombs-1):
        for rhomb_j in range(1, numRhombs): 
            if rhomb_i != rhomb_j:
               
                DL_i = np.column_stack(( dip_centers[4*(rhomb_i) : 4*(rhomb_i)+4, :], D_l_vecs[4*(rhomb_i) : 4*(rhomb_i)+4, :] ))
                DS_i = np.column_stack(( dip_centers[4*(rhomb_i) : 4*(rhomb_i)+4, :], D_s_vecs[4*(rhomb_i) : 4*(rhomb_i)+4, :] ))
                Q_i  = np.column_stack(( dip_centers[4*(rhomb_i) : 4*(rhomb_i)+4, :], Q_vecs[4*(rhomb_i) : 4*(rhomb_i)+4, :] ))

                DL_j = np.column_stack(( dip_centers[4*(rhomb_j) : 4*(rhomb_j)+4, :], D_l_vecs[4*(rhomb_j) : 4*(rhomb_j)+4, :] ))
                DS_j = np.column_stack(( dip_centers[4*(rhomb_j) : 4*(rhomb_j)+4, :], D_s_vecs[4*(rhomb_j) : 4*(rhomb_j)+4, :] ))
                Q_j  = np.column_stack(( dip_centers[4*(rhomb_j) : 4*(rhomb_j)+4, :], Q_vecs[4*(rhomb_j) : 4*(rhomb_j)+4, :] ))

                ### DL_i coupled with (DL_1, DS_1, Q_1, ... DL_N, DS_N, Q_N) ### 
                H[3*rhomb_i, 3*rhomb_j]   = make_g(mode_n=DL_i, mode_m=DL_j, m=m_DL, k=k) / eigval_thisround
                H[3*rhomb_i, 3*rhomb_j+1] = make_g(mode_n=DL_i, mode_m=DS_j, m=m_DL, k=k) / eigval_thisround
                H[3*rhomb_i, 3*rhomb_j+2] = make_g(mode_n=DL_i, mode_m=Q_j, m=m_DL, k=k) / eigval_thisround
                
                ### DS_i coupled with (DL_1, DS_1, Q_1, ... DL_N, DS_N, Q_N) ###
                H[3*rhomb_i+1,3*rhomb_j]   = make_g(mode_n=DS_i, mode_m=DL_j, m=m_DS, k=k) / eigval_thisround
                H[3*rhomb_i+1,3*rhomb_j+1] = make_g(mode_n=DS_i, mode_m=DS_j, m=m_DS, k=k) / eigval_thisround
                H[3*rhomb_i+1,3*rhomb_j+2] = make_g(mode_n=DS_i, mode_m=Q_j, m=m_DS, k=k) / eigval_thisround

                ### Q_i coupled with (DL_1, DS_1, Q_1, ... DL_N, DS_N, Q_N) ###
                H[3*rhomb_i+2,3*rhomb_j]   = make_g(mode_n=Q_i, mode_m=DL_j, m=m_Q, k=k) / eigval_thisround
                H[3*rhomb_i+2,3*rhomb_j+1] = make_g(mode_n=Q_i, mode_m=DS_j, m=m_Q, k=k) / eigval_thisround
                H[3*rhomb_i+2,3*rhomb_j+2] = make_g(mode_n=Q_i, mode_m=Q_j, m=m_Q, k=k) / eigval_thisround

                ########## Now the opposite of the above terms ##########

                H[3*rhomb_j, 3*rhomb_i]   = make_g(mode_n=DL_j, mode_m=DL_i, m=m_DL, k=k) / eigval_thisround
                H[3*rhomb_j+1, 3*rhomb_i] = make_g(mode_n=DS_j, mode_m=DL_i, m=m_DS, k=k) / eigval_thisround
                H[3*rhomb_j+2, 3*rhomb_i] = make_g(mode_n=Q_j, mode_m=DL_i, m=m_Q, k=k) / eigval_thisround
                
                H[3*rhomb_j,3*rhomb_i+1]   = make_g(mode_n=DL_j, mode_m=DS_i, m=m_DL, k=k) / eigval_thisround
                H[3*rhomb_j+1,3*rhomb_i+1] = make_g(mode_n=DS_j, mode_m=DS_i, m=m_DS, k=k) / eigval_thisround
                H[3*rhomb_j+2,3*rhomb_i+1] = make_g(mode_n=Q_j, mode_m=DS_i, m=m_Q, k=k) / eigval_thisround

                H[3*rhomb_j,3*rhomb_i+2]   = make_g(mode_n=DL_j, mode_m=Q_i, m=m_DL, k=k) / eigval_thisround
                H[3*rhomb_j+1,3*rhomb_i+2] = make_g(mode_n=DS_j, mode_m=Q_i, m=m_DS, k=k) / eigval_thisround
                H[3*rhomb_j+2,3*rhomb_i+2] = make_g(mode_n=Q_j, mode_m=Q_i, m=m_Q, k=k) / eigval_thisround
        H = H*hbar_eVs**2
        eigval, eigvec = np.linalg.eig(H)
    return np.real(eigval), eigvec, H

def interate():
    final_eigvals = np.zeros(np.int(3*numRhombs))
    final_eigvecs = np.zeros( (np.int(3*numRhombs),np.int(3*numRhombs)),dtype=complex) 

    for mode in range(0,np.int(3*numRhombs)): #converge each mode individually 
        eigval_hist = np.array([Q_wsp, Q_wsp*0.5],dtype=np.double)*hbar_eVs
        eigvec_hist = np.column_stack((np.zeros((np.int(3*numRhombs)),dtype=complex), 1+np.zeros((np.int(3*numRhombs)),dtype=complex)))

        count = 0
        while (np.abs((eigval_hist[0] - eigval_hist[1]))  > 10**(-prec)) and (np.linalg.norm((np.abs(eigvec_hist[:,0] - eigvec_hist[:,1])) > 10**(-prec))):
            eigval_thisround = eigval_hist[0]/hbar_eVs

            if count > 100: 
                denom = ( eigval_hist[2] - eigval_hist[1] ) - ( eigval_hist[1] - eigval_hist[0] )
                eigval_thisround = eigval_hist[2] - ( eigval_hist[2] - eigval_hist[1] )**2 / denom / hbar_eVs

            k = (eigval_thisround*np.sqrt(eps_b))/c
            w,v,H = make_H(k=k)
            new_eigvals = w[mode]
            eigval_hist = np.append(new_eigvals, eigval_hist)

            if count == 6000: 
                print('THIS CODE DID NOT CONVERGE GURL </3'); sys.exit()
            count = count + 1

        final_eigvals[mode] = eigval_hist[0]
        final_eigvecs[:,mode] = v[:,mode]

    final_eigvecs = final_eigvecs.transpose() #now the rows correspond to the ith mode v[i,:]
    total = np.column_stack((final_eigvals,final_eigvecs)) #arrange a new matrix, where the first column is the evals and the remaining columns are evectors
    total = total[np.argsort(total[:,0])]
    return total

interate()