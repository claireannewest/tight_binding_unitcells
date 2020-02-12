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

inputs = np.loadtxt(param['inputs'],skiprows=1)

rhomb_centers = inputs[:,0:2]*1E-7
dip_centers = inputs[:,2:4]*1E-7
D_l_vecs = inputs[:,4:6]
D_s_vecs = inputs[:,6:8]
Q_vecs = inputs[:,8:10]
numRhombs = len(inputs)/4

w0_DL = 1.30/hbar_eVs
w0_DS = 1.62/hbar_eVs
w0_Q = 1.80/hbar_eVs

m_DL = 2.48E-34
m_DS = 3.02E-34
m_Q = 1.25E-33

gamNR_DL = 0.45 /hbar_eVs
gamNR_DS = 0.74 / hbar_eVs
gamNR_Q = 0.16 / hbar_eVs

def DL(i): #allows me to grab the 4 dipole centers and directions for each rhombus
    dipcent_i = dip_centers[4*i : 4*i+4, :]
    vecs = D_l_vecs[4*i : 4*i+4, :]
    return np.column_stack(( dipcent_i, vecs ))

def DS(i): #allows me to grab the 4 dipole centers and directions for each rhombus
    dipcent_i = dip_centers[4*i : 4*i+4, :]
    vecs = D_s_vecs[4*i : 4*i+4, :]
    return np.column_stack(( dipcent_i, vecs ))

def Q(i): #allows me to grab the 4 dipole centers and directions for each rhombus
    dipcent_i = dip_centers[4*i : 4*i+4, :]
    vecs = Q_vecs[4*i : 4*i+4, :]
    return np.column_stack(( dipcent_i, vecs ))

def make_g(mode_i, mode_j,m,k): #mode 1,2 are four columns: [sph_cent_x, sph_cent_y, vec_x, vec_y] and four rows corresponding to the four spheres
# i, j are the indices of the rhombii. n,m are the indices of the dipoles.
    gtot=0
    for sph_n in range(0,4):
        for sph_m in range(0,4):
            r_nm = mode_i[sph_n,0:2]-mode_j[sph_m,0:2] #distance between the nth and mth sphere
            mag_rnm = np.linalg.norm(r_nm)
            nhat_nm = r_nm / mag_rnm
            xn_hat = mode_i[sph_n, 2:4]/np.linalg.norm(mode_i[sph_n, 2:4])
            xm_hat = mode_j[sph_m, 2:4]/np.linalg.norm(mode_j[sph_m, 2:4])
            xn_dot_nn_dot_xm = np.dot(xn_hat, nhat_nm)*np.dot(nhat_nm, xm_hat)
            nearField = ( 3.*xn_dot_nn_dot_xm - np.dot(xn_hat,xm_hat) ) / mag_rnm**3
    #         #intermedField = 1j*k*(3*xn_dot_nn_dot_xm - np.dot(xn_hat,xn_hat)) / mag_rnm**2 
    #         #farField = k**2*(xn_dot_nn_dot_xm - np.dot(xn_hat,xm_hat)) / mag_rnm
            g_ind = e**2 * ( nearField )# - intermedField - farField ) * np.exp(1j*k*mag_rnm)  
            gtot=g_ind+gtot
    return 0#-gtot/(2*m)

#print make_g(mode_i=Q(i=1), mode_j=DL(i=0),m=m_DL,k=1)



def make_H(k):
    H = np.zeros( (np.int(3*numRhombs),np.int(3*numRhombs)),dtype=complex) 
    #w_thisround = k*c/np.sqrt(eps_b) #1/s

    gam_DL = gamNR_DL #+ (w_thisround)**2*(2.0*e**2)/(3.0*m_DL*c**3)
    gam_DS = gamNR_DS# + (w_thisround)**2*(2.0*e**2)/(3.0*m_DS*c**3)
    gam_Q = gamNR_Q #+ (w_thisround)**2*(2.0*e**2)/(3.0*m_Q*c**3)

    for i in range(0, numRhombs): #handle the on diagonal terms 
        #print w_thisround
        H[3*i,3*i]     = w0_DL**2#/w_thisround- 1j*gam_DL
        H[3*i+1,3*i+1] = w0_DS**2#/w_thisround - 1j*gam_DS
        H[3*i+2,3*i+2] = w0_Q**2#/w_thisround - 1j*gam_Q

    for rhomb_i in range(0 , numRhombs-1):
        for rhomb_j in range(1, numRhombs): 
            if rhomb_i != rhomb_j:
               
                ### DL_i coupled with (DL_1, DS_1, Q_1, ... DL_N, DS_N, Q_N) ### 
                H[3*rhomb_i, 3*rhomb_j]   = make_g(mode_i=DL(i=rhomb_i), mode_j=DL(i=rhomb_j), m=m_DL, k=k) #/ w_thisround
                H[3*rhomb_i, 3*rhomb_j+1] = make_g(mode_i=DL(i=rhomb_i), mode_j=DS(i=rhomb_j), m=m_DL, k=k) #/ w_thisround
                H[3*rhomb_i, 3*rhomb_j+2] = make_g(mode_i=DL(i=rhomb_i), mode_j=Q(i=rhomb_j), m=m_DL, k=k) #/ w_thisround
                
                ### DS_i coupled with (DL_1, DS_1, Q_1, ... DL_N, DS_N, Q_N) ###
                H[3*rhomb_i+1,3*rhomb_j]   = make_g(mode_i=DS(i=rhomb_i), mode_j=DL(i=rhomb_j), m=m_DS, k=k)# / w_thisround
                H[3*rhomb_i+1,3*rhomb_j+1] = make_g(mode_i=DS(i=rhomb_i), mode_j=DS(i=rhomb_j), m=m_DS, k=k)# / w_thisround
                H[3*rhomb_i+1,3*rhomb_j+2] = make_g(mode_i=DS(i=rhomb_i), mode_j=Q(i=rhomb_j), m=m_DS, k=k) #/ w_thisround

                ### Q_i coupled with (DL_1, DS_1, Q_1, ... DL_N, DS_N, Q_N) ###
                H[3*rhomb_i+2,3*rhomb_j]   = make_g(mode_i=Q(i=rhomb_i), mode_j=DL(i=rhomb_j), m=m_Q, k=k)# / w_thisround
                H[3*rhomb_i+2,3*rhomb_j+1] = make_g(mode_i=Q(i=rhomb_i), mode_j=DS(i=rhomb_j), m=m_Q, k=k)# / w_thisround
                H[3*rhomb_i+2,3*rhomb_j+2] = make_g(mode_i=Q(i=rhomb_i), mode_j=Q(i=rhomb_j), m=m_Q, k=k) #/ w_thisround

                ########## Now the opposite of the above terms ##########

                H[3*rhomb_j, 3*rhomb_i]   = make_g(mode_i=DL(i=rhomb_j), mode_j=DL(i=rhomb_i), m=m_DL, k=k)# / w_thisround
                H[3*rhomb_j+1, 3*rhomb_i] = make_g(mode_i=DS(i=rhomb_j), mode_j=DL(i=rhomb_i), m=m_DS, k=k)# / w_thisround
                H[3*rhomb_j+2, 3*rhomb_i] = make_g(mode_i=Q(i=rhomb_j), mode_j=DL(i=rhomb_i), m=m_Q, k=k)# / w_thisround
                
                H[3*rhomb_j,3*rhomb_i+1]   = make_g(mode_i=DL(i=rhomb_j), mode_j=DS(i=rhomb_i), m=m_DL, k=k) #/ w_thisround
                H[3*rhomb_j+1,3*rhomb_i+1] = make_g(mode_i=DS(i=rhomb_j), mode_j=DS(i=rhomb_i), m=m_DS, k=k) #/ w_thisround
                H[3*rhomb_j+2,3*rhomb_i+1] = make_g(mode_i=Q(i=rhomb_j), mode_j=DS(i=rhomb_i), m=m_Q, k=k)# / w_thisround

                H[3*rhomb_j,3*rhomb_i+2]   = make_g(mode_i=DL(i=rhomb_j), mode_j=Q(i=rhomb_i), m=m_DL, k=k)# / w_thisround
                H[3*rhomb_j+1,3*rhomb_i+2] = make_g(mode_i=DS(i=rhomb_j), mode_j=Q(i=rhomb_i), m=m_DS, k=k)# / w_thisround
                H[3*rhomb_j+2,3*rhomb_i+2] = make_g(mode_i=Q(i=rhomb_j), mode_j=Q(i=rhomb_i), m=m_Q, k=k) #/ w_thisround
        eigval, eigvec = np.linalg.eig(H)
    return eigval, eigvec, H

def interate():
    final_eigvals = np.zeros(np.int(3*numRhombs),dtype=complex)
    final_eigvecs = np.zeros( (np.int(3*numRhombs),np.int(3*numRhombs)),dtype=complex) 

    for mode in range(0,np.int(3*numRhombs)): #converge each mode individually 
        eigval_hist = np.array([w0_Q, w0_Q*0.7],dtype=complex)
        eigvec_hist = np.column_stack((np.zeros((np.int(3*numRhombs)),dtype=complex), 1+np.zeros((np.int(3*numRhombs)),dtype=complex)))
        count = 0
        while (np.abs((eigval_hist[0] - eigval_hist[1])*hbar_eVs)  > 10**(-prec)):# and (np.linalg.norm((np.abs(eigvec_hist[:,0] - eigvec_hist[:,1])) > 10**(-prec))):
            w_thisround = np.real(eigval_hist[0])
            #print mode, count, '%.2E' %w_thisround
            if count > 100: 
                denom = ( eigval_hist[2] - eigval_hist[1] ) - ( eigval_hist[1] - eigval_hist[0] )
                w_thisround = np.real(eigval_hist[2] - ( eigval_hist[2] - eigval_hist[1] )**2 / denom )
            if w_thisround == 0:
                w_thisround = np.real(eigval_hist[0])

            k = (w_thisround*np.sqrt(eps_b))/c
            w,v,H = make_H(k=k)
            new_eigvals = w[mode]#np.sqrt((w[mode]))
            eigval_hist = np.append(new_eigvals, eigval_hist)

            if count == 6000: 
                print('THIS CODE DID NOT CONVERGE GURL </3'); sys.exit()
            count = count + 1            
            #print mode, eigval_hist[0]

        final_eigvals[mode] = eigval_hist[0]
        final_eigvecs[:,mode] = v[:,mode]

    final_eigvecs = final_eigvecs.transpose() #now the rows correspond to the ith mode v[i,:]
    total = np.column_stack((final_eigvals,final_eigvecs)) #arrange a new matrix, where the first column is the evals and the remaining columns are evectors
    total = total[np.argsort(total[:,0])]
    return total

interate()


