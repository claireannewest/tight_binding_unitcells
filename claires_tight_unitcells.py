import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import math
import yaml
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
D_l_vecs = inputs[:,4:6]*1E-7
D_s_vecs = inputs[:,6:8]*1E-7
Q_vecs    = inputs[:,8:10]*1E-7

numRhombs = len(inputs)/4

numIndModes = 3

w0_DL = 1.34#1.43
w0_DS = 1.63#1.81
w0_Q = 1.78

m_DL = 1.3E-33 #1.016E-34
m_DS = .79E-33 #5.079E-35
m_Q  = 1.3E-33 #3.611E-34

gamNR_DL = 0.07#45
gamNR_DS = 0.07#8
gamNR_Q = 0.07#16

def DL(i): #allows me to grab the 4 dipole centers and directions for each rhombus
    rhomb_cent_i = rhomb_centers[4*i : 4*i+4, :]
    dipcent_i = dip_centers[4*i : 4*i+4, :]
    vecs = D_l_vecs[4*i : 4*i+4, :]
    return np.column_stack(( rhomb_cent_i, dipcent_i, vecs ))


def DS(i): #allows me to grab the 4 dipole centers and directions for each rhombus
    rhomb_cent_i = rhomb_centers[4*i : 4*i+4, :]
    dipcent_i = dip_centers[4*i : 4*i+4, :]
    vecs = D_s_vecs[4*i : 4*i+4, :]
    return np.column_stack(( rhomb_cent_i, dipcent_i, vecs ))

def Q(i): #allows me to grab the 4 dipole centers and directions for each rhombus
    rhomb_cent_i = rhomb_centers[4*i : 4*i+4, :]
    dipcent_i = dip_centers[4*i : 4*i+4, :]
    vecs = Q_vecs[4*i : 4*i+4, :]
    return np.column_stack(( rhomb_cent_i, dipcent_i, vecs ))

def make_g(mode_i, mode_j,m,k): #mode 1,2 are four columns: [sph_cent_x, sph_cent_y, vec_x, vec_y] and four rows corresponding to the four spheres
# i, j are the indices of the rhombii. n,m are the indices of the dipoles.
    k = np.real(k)
    g_ind = np.zeros((16),dtype=complex)
    xmag = np.zeros((16))
    gtot = 0
    for sph_n in range(0,4):
        for sph_m in range(0,4):
            #r_nm = mode_i[sph_n,0:2] - mode_j[sph_m,0:2] #distance between the ith and jth rhomb
            r_nm = mode_i[sph_n,2:4]-mode_j[sph_m,2:4]  #distance between the nth and mth sphere
            mag_rnm = np.linalg.norm(r_nm)
            nhat_nm = r_nm / mag_rnm
            if np.linalg.norm(mode_i[sph_n, 4:6]) == 0:
                g_ind = 0; break
            if np.linalg.norm(mode_j[sph_m, 4:6]) == 0:
                g_ind = 0; break   

            xn = mode_i[sph_n, 4:6]  
            xm = mode_j[sph_m, 4:6]   
            xn_hat = xn/np.linalg.norm(xn)
            xm_hat = xm/np.linalg.norm(xm)

            xn_dot_nn_dot_xm = np.dot(xn_hat, nhat_nm)*np.dot(nhat_nm, xm_hat)
            #print 3.*xn_dot_nn_dot_xm - np.dot(xn_hat,xm_hat)
            nearField = ( 3.*xn_dot_nn_dot_xm - np.dot(xn_hat,xm_hat) ) / mag_rnm**3
            intermedField = 1j*k*(3*xn_dot_nn_dot_xm - np.dot(xn_hat,xm_hat)) / mag_rnm**2 
            farField = k**2*(xn_dot_nn_dot_xm - np.dot(xn_hat,xm_hat)) / mag_rnm
            g =  e**2 * hbar_eVs**2 * ( nearField - intermedField - farField ) * np.exp(1j*k*mag_rnm) 
            g_ind[sph_m+4*sph_n] = g
            xmag[sph_m+4*sph_n] = np.linalg.norm(xn)

            #print sph_n, sph_m, g_ind
            #gtot = g + gtot
    avg_arrow_len = np.sum(xmag)/len(xmag)
    weight = xmag/avg_arrow_len
    g_weight = np.dot(g_ind, weight)/len(g_ind)

    return -g_weight/(m)

#make_g(mode_i=DS(i=0), mode_j=DS(i=1), m=m_Q, k=1.51/hbar_eVs/c)
# print g_ind[3]
# print xmag[:,3]


def make_H(k):
    H = np.zeros( (np.int(numIndModes*numRhombs),np.int(numIndModes*numRhombs)),dtype=complex) 
    w_thisround = k*c/np.sqrt(eps_b)*hbar_eVs #eV

    gam_DL = gamNR_DL + (np.real(w_thisround))**2*(2.0*e**2)/(3.0*m_DL*c**3)/hbar_eVs
    gam_DS = gamNR_DS + (np.real(w_thisround))**2*(2.0*e**2)/(3.0*m_DS*c**3)/hbar_eVs
    gam_Q  = gamNR_Q  + (np.real(w_thisround))**2*(2.0*e**2)/(3.0*m_Q*c**3)/hbar_eVs
    print (np.real(w_thisround))**2*(2.0*e**2)/(3.0*m_Q*c**3)/hbar_eVs
    print ((w_thisround))**2*(2.0*e**2)/(3.0*m_Q*c**3)/hbar_eVs
    for i in range(0, numRhombs): #handle the on diagonal terms 
        H[3*i,3*i]     = w0_DL**2 - 1j*gam_DL*w_thisround
        H[3*i+1,3*i+1] = w0_DS**2 - 1j*gam_DS*w_thisround
        H[3*i+2,3*i+2] = w0_Q**2 - 1j*gam_Q*w_thisround

    for rhomb_i in range(0 , numRhombs-1):
        for rhomb_j in range(1, numRhombs): 
            if rhomb_i != rhomb_j:
               
                ### DL_i coupled with (DL_1, DS_1, Q_1, ... DL_N, DS_N, Q_N) ### 
                H[3*rhomb_i, 3*rhomb_j]   = make_g(mode_i=DL(i=rhomb_i), mode_j=DL(i=rhomb_j), m=m_DL, k=k)
                H[3*rhomb_i, 3*rhomb_j+1] = make_g(mode_i=DL(i=rhomb_i), mode_j=DS(i=rhomb_j), m=m_DL, k=k)
                H[3*rhomb_i, 3*rhomb_j+2] = make_g(mode_i=DL(i=rhomb_i), mode_j=Q(i=rhomb_j), m=m_DL, k=k)
                
                ### DS_i coupled with (DL_1, DS_1, Q_1, ... DL_N, DS_N, Q_N) ###
                H[3*rhomb_i+1,3*rhomb_j]   = make_g(mode_i=DS(i=rhomb_i), mode_j=DL(i=rhomb_j), m=m_DS, k=k)
                H[3*rhomb_i+1,3*rhomb_j+1] = make_g(mode_i=DS(i=rhomb_i), mode_j=DS(i=rhomb_j), m=m_DS, k=k)
                H[3*rhomb_i+1,3*rhomb_j+2] = make_g(mode_i=DS(i=rhomb_i), mode_j=Q(i=rhomb_j), m=m_DS, k=k)

                ### Q_i coupled with (DL_1, DS_1, Q_1, ... DL_N, DS_N, Q_N) ###
                H[3*rhomb_i+2,3*rhomb_j]   = make_g(mode_i=Q(i=rhomb_i), mode_j=DL(i=rhomb_j), m=m_Q, k=k)
                H[3*rhomb_i+2,3*rhomb_j+1] = make_g(mode_i=Q(i=rhomb_i), mode_j=DS(i=rhomb_j), m=m_Q, k=k)
                H[3*rhomb_i+2,3*rhomb_j+2] = make_g(mode_i=Q(i=rhomb_i), mode_j=Q(i=rhomb_j), m=m_Q, k=k)

                ########## Now the opposite of the above terms ##########
                H[3*rhomb_j, 3*rhomb_i]   = make_g(mode_i=DL(i=rhomb_j), mode_j=DL(i=rhomb_i), m=m_DL, k=k)
                H[3*rhomb_j+1, 3*rhomb_i] = make_g(mode_i=DS(i=rhomb_j), mode_j=DL(i=rhomb_i), m=m_DS, k=k)
                H[3*rhomb_j+2, 3*rhomb_i] = make_g(mode_i=Q(i=rhomb_j), mode_j=DL(i=rhomb_i), m=m_Q, k=k)
                
                H[3*rhomb_j,3*rhomb_i+1]   = make_g(mode_i=DL(i=rhomb_j), mode_j=DS(i=rhomb_i), m=m_DL, k=k)
                H[3*rhomb_j+1,3*rhomb_i+1] = make_g(mode_i=DS(i=rhomb_j), mode_j=DS(i=rhomb_i), m=m_DS, k=k)
                H[3*rhomb_j+2,3*rhomb_i+1] = make_g(mode_i=Q(i=rhomb_j), mode_j=DS(i=rhomb_i), m=m_Q, k=k)

                H[3*rhomb_j,3*rhomb_i+2]   = make_g(mode_i=DL(i=rhomb_j), mode_j=Q(i=rhomb_i), m=m_DL, k=k)
                H[3*rhomb_j+1,3*rhomb_i+2] = make_g(mode_i=DS(i=rhomb_j), mode_j=Q(i=rhomb_i), m=m_DS, k=k)
                H[3*rhomb_j+2,3*rhomb_i+2] = make_g(mode_i=Q(i=rhomb_j), mode_j=Q(i=rhomb_i), m=m_Q, k=k)

        eigval, eigvec = np.linalg.eig(H)

    return eigval, eigvec, H

def interate():
    final_eigvals = np.zeros(np.int(3*numRhombs),dtype=complex)
    final_eigvecs = np.zeros( (np.int(3*numRhombs), np.int(3*numRhombs)), dtype=complex) 
    w_DLstart = -1j*gamNR_DL/2. + np.sqrt(-gamNR_DL**2/4.+w0_DL**2)
    w_DSstart = -1j*gamNR_DS/2. + np.sqrt(-gamNR_DS**2/4.+w0_DS**2)
    w_Qstart = -1j*gamNR_Q/2. + np.sqrt(-gamNR_Q**2/4.+w0_Q**2)

    for mode in range(0,np.int(3*numRhombs)): #converge each mode individually         
        if mode == 0 or mode == 3 or mode == 6: 
            eigval_hist = np.array([w_DLstart, w_DLstart*1.1],dtype=complex) 
        if mode == 1 or mode == 4 or mode == 7:
            eigval_hist = np.array([w_DSstart, w_DSstart*1.2],dtype=complex) 
        if mode == 2 or mode == 5 or mode == 8:
            eigval_hist = np.array([w_Qstart, w_Qstart*1.2],dtype=complex) 

        eigvec_hist = np.zeros((3*numRhombs, 2))
        eigvec_hist[:,0] = 1.
        vec_prec = np.zeros((3*numRhombs, 1))+10**(-prec)

        count = 0
        inercount = 1

        while np.abs((np.real(eigval_hist[0]) - np.real(eigval_hist[1])))  > 10**(-prec) and np.sum(np.abs((eigvec_hist[:,0] - eigvec_hist[:,1]))) > 10**(-prec):
            w_thisround = eigval_hist[0]
            
            if count > 100: 
               denom = ( eigval_hist[2] - eigval_hist[1] ) - ( eigval_hist[1] - eigval_hist[0] )
               w_thisround = eigval_hist[2] - ( eigval_hist[2] - eigval_hist[1] )**2 / denom 
            
            k = w_thisround/hbar_eVs*np.sqrt(eps_b)/c

            val, vec, H = make_H(k=k)

            amp = np.sqrt(np.abs(val))
            phi = np.arctan2(np.imag(val), np.real(val))
            energy = amp*np.cos(phi/2)

            post_sort_val = energy[energy.argsort()]
            post_sort_vec = vec[:,energy.argsort()]

            this_val = post_sort_val[mode]
            this_vec = post_sort_vec[:,mode]
            new_eigvals = this_val

            eigval_hist = np.append(new_eigvals, eigval_hist)
            eigvec_hist = np.column_stack((this_vec, eigvec_hist))

            print mode, count

            count = count + 1 

        final_eigvals[mode] = eigval_hist[0]
        final_eigvecs[:,mode] = eigvec_hist[:,0]
    return final_eigvals, final_eigvecs

D_l_vecs = inputs[:,4:6]
D_s_vecs = inputs[:,6:8] 
Q_vecs = inputs[:,8:10]

def seeVectors(mode):
    w = np.real(final_eigvals[mode])
    v = np.real(final_eigvecs[:,mode])

    sph_ycoords = dip_centers[:,0]
    sph_zcoords = dip_centers[:,1]  
   
    plt.subplot(1,3*numRhombs,mode+1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.title('%.2f eV' % (w), fontsize=18)
    plt.scatter(sph_ycoords, sph_zcoords,c='blue',s=50)

    for rhomb_i in range(0, numRhombs):
        DL_i = D_l_vecs[4*(rhomb_i) : 4*(rhomb_i)+4, :]
        DS_i = D_s_vecs[4*(rhomb_i) : 4*(rhomb_i)+4, :]
        Q_i  = Q_vecs[4*(rhomb_i) : 4*(rhomb_i)+4, :]
        mag_mode = (v[3*rhomb_i]*DL_i + v[3*rhomb_i+1]*DS_i +v[3*rhomb_i+2]*Q_i)

        ymin = min(sph_ycoords)-1E-5; ymax = max(sph_ycoords)+1E-5
        zmin = min(sph_zcoords)-1E-5; zmax = max(sph_zcoords)+1E-5
        plt.quiver(sph_ycoords[4*rhomb_i : 4*rhomb_i+4], sph_zcoords[4*rhomb_i : 4*rhomb_i+4], mag_mode[:,0], mag_mode[:,1], pivot='mid', 
            width=.5, #shaft width in arrow units 
            scale=1., 
            headlength=5,
            headwidth=5.,#5.8
            minshaft=4., #4.1
            minlength=.1)
    plt.xlim([ymin, ymax])
    plt.ylim([zmin, zmax])
    plt.yticks([])
    plt.xticks([])
    #plt.show()
    return w, mag_mode

final_eigvals, final_eigvecs = interate()
print final_eigvals[1], np.real(final_eigvecs[:,1])
print final_eigvals[2], np.real(final_eigvecs[:,2])


fig = plt.figure(num=None, figsize=(12, 3), dpi=80, facecolor='w', edgecolor='k')   
for mode in range(0,3*numRhombs):
    seeVectors(mode=mode)
plt.show()
