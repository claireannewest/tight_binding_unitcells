import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sys
from decimal import Decimal
import math
import yaml
from scipy import optimize
from scipy.special import kn

open_param_file = open('parameter_files/parameters.yaml')
param = yaml.load(open_param_file)
c = param['constants']['c']
eps_b = 1.0
hbar_eVs = param['constants']['hbar_eVs']
e = param['constants']['e']
prec = param['precision']
hbar_cgs = param['constants']['hbar_cgs']

inputdata = np.loadtxt('inputs.txt', skiprows=1)
n = 1.
nm_to_per_s = 2*np.pi*c/(n)*1e7 # omega = this variable / lambda (in nm)
um_to_per_s = 2*np.pi*c/(n)*1e4 # omega = this variable / lambda (in um)
dip_per_part = 2
dip_origins = inputdata[:,2:4]*1E-7
numDips = len(dip_origins)
dip_dirs = inputdata[:,4:6]*1E-7

### The name of the game is Gam_eels \propto Im[alpha]. I need an expression for 
### total alpha, which is alpha_tot = eE_tot/x_tot 

def loadData():
    data_long = np.loadtxt('/Users/clairewest/werk/research/penrose/my_sims/monomer/Spectrum_2DS_long',skiprows=1)
    w_long = um_to_per_s / data_long[:,0] # needs to be in 1/s
    effic_long = data_long[:,2]
    allData_long = np.column_stack([w_long, effic_long])
    allData_sortlong = allData_long[allData_long[:,0].argsort()[::-1]]
    idx = np.where(allData_sortlong[:,0] >= 1.6/hbar_eVs)     
    allData_sortlong = np.delete(allData_sortlong, idx, axis=0)
    idx = np.where(allData_sortlong[:,0] <= 1./hbar_eVs)  
    allData_sortlong = np.delete(allData_sortlong, idx, axis=0)


    w_long = np.asarray(allData_sortlong[:,0]) 
    effic_sim_long = np.asarray(allData_sortlong[:,1])/max(allData_sortlong[:,1]) 

    data_short = np.loadtxt('/Users/clairewest/werk/research/penrose/my_sims/monomer/Spectrum_2DS_short',skiprows=1)
    w_short = um_to_per_s / data_short[:,0] # needs to be in 1/s
    effic_short = data_short[:,2]
    allData_short = np.column_stack([w_short, effic_short])
    allData_sortshort = allData_short[allData_short[:,0].argsort()[::-1]]

    idx = np.where(allData_sortshort[:,0] >= 1.6/hbar_eVs)     
    allData_sortshort = np.delete(allData_sortshort, idx, axis=0)
    idx = np.where(allData_sortshort[:,0] <= 1./hbar_eVs)  
    allData_sortshort = np.delete(allData_sortshort, idx, axis=0)

    w_short = np.asarray(allData_sortshort[:,0]) 
    effic_sim_short = np.asarray(allData_sortshort[:,1])/max(allData_sortshort[:,1])   
    
    w = np.concatenate((w_long, w_short),axis=0)
    #effic_sim = np.concatenate((effic_sim_long, effic_sim_short),axis=0)
    effic_sim = effic_sim_long
    w = w_long
    return [w, effic_sim]


# cs=170./2*1E-7
# a=35./2*1E-7

def prolate_parameters(
        cs, # semi-major axis, units of cm
        a,   # semi-minor axis, units of cm
        which # which dipole excitation 
        ): 
    es = (cs**2 - a**2)/cs**2
    Lz = (1-es**2)/es**3*(-es+1./2*np.log((1+es)/(1-es)))
    Ly = (1-Lz)/2   
    D = 3./4*((1+es**2)/(1-es**2)*Lz + 1)
    V = 4./3*np.pi*a**2*cs
    if which == 'long': li = cs; Li = Lz
    if which == 'short': li = a; Li = Ly
    return D,li, Li, V

def make_dip_masses( # calculates the mass, damping, resonance frequency of a spheriod in modified long wavelength approx.
        w, # 1/s
        w0, # eV
        gamNR, # eV
        which,
        cs,
        a): 
    D, li, Li, V = prolate_parameters(cs=cs, a=a, which=which)
    eps_inf = 9.
    m = 4*np.pi*e**2*((eps_inf-1)+1/Li)/((w0/hbar_eVs)**2*V/Li**2) # g 
    m_LW = m + D*e**2/(li*c**2) # g (charge and speed of light)
    w0_LW = (w0/hbar_eVs)*np.sqrt(m/m_LW) # 1/s
    gam_LW = gamNR/hbar_eVs*(m/m_LW) + 2*e**2/(3*m_LW*c**3)*w**2 # 1/s
    return m_LW, w0_LW, gam_LW # [m_LW]=g, [w0_LW] = 1/s, [gam_LW] = 1/s

#m_LW, w0_LW, gam_LW = make_dip_masses(w=loadData()[0], w0=1., gamNR=0.1,which='short')

def make_g( # calculates the dipole-dipole coupling between i and j
    w_all, #1/s
    i, # integer, dipole i
    j, # integer, dipole j
    m # mass in g
    ):
    xi_hat = dip_dirs[i,:]/np.linalg.norm(dip_dirs[i,:])
    xj_hat = dip_dirs[j,:]/np.linalg.norm(dip_dirs[j,:])
    r_ij = dip_origins[i,:]-dip_origins[j,:]
    mag_rij = np.linalg.norm(r_ij)
    if mag_rij == 0:
        g=0
    else:
        nhat_ij = r_ij / mag_rij

        theta_xi_nij = np.arctan2(xi_hat[1]-nhat_ij[1], xi_hat[0]-nhat_ij[0])
        if theta_xi_nij<0:
            theta_xi_nij = np.pi+theta_xi_nij

        theta_xj_nij = np.arctan2(xj_hat[1]-nhat_ij[1], xj_hat[0]-nhat_ij[0])
        if theta_xj_nij<0:
            theta_xj_nij = np.pi+theta_xj_nij

        theta_xi_xj = np.arctan2(xi_hat[1]-xj_hat[1], xi_hat[0]-xj_hat[0])
        if theta_xi_xj<0:
            theta_xi_xj = np.pi+theta_xi_xj

        xi_dot_nn_dot_xj = np.cos(theta_xi_nij)*np.cos(theta_xj_nij)

        k = w_all*np.sqrt(eps_b)/c
        nearField = ( 3.*xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat) ) / mag_rij**3
        intermedField = 1j*k*(3*xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat)) / mag_rij**2 
        farField = k**2*(xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat)) / mag_rij
        g = e**2 * ( nearField - intermedField - farField ) * np.exp(1j*k*mag_rij)  
    return g # units of g/s^2

#print make_g(w_all=loadData()[0],i=0,j=2, m=1E-33)



def make_invH( ### use this to calculate alpha
    w_all, # each omega fills the first dimension of array
    wsp_long, 
    wsp_short,
    gamNR_long, 
    gamNR_short,
    cs,
    a
    ):
    H = np.zeros( (len(w_all), np.int(numDips),np.int(numDips)),dtype=complex) 
    m_LW_long, w0_LW_long, gam_LW_long = make_dip_masses(w=w_all, w0=wsp_long, gamNR=gamNR_long,which='long',cs=cs, a=a) #units are g, 1/s, 1/s
    m_LW_short, w0_LW_short, gam_LW_short = make_dip_masses(w=w_all, w0=wsp_short, gamNR=gamNR_short,which='short',cs=cs, a=a) #units are g, 1/s, 1/s
    
    # vary w, then on diagonal = wsp, off diagnol = 0
    dipole_wsps = np.zeros( (np.int(numDips),np.int(numDips)))
    dipole_wsps[:numDips/2,:numDips/2] = w0_LW_long*np.identity(numDips/2)
    dipole_wsps[numDips/2:,numDips/2:] = w0_LW_short*np.identity(numDips/2)
    all_wsps = np.repeat(dipole_wsps[np.newaxis,:, :], len(w_all), axis=0)

    dipole_masses = np.zeros( (np.int(numDips),np.int(numDips)))
    dipole_masses[:numDips/2,:numDips/2] = m_LW_long*np.identity(numDips/2)
    dipole_masses[numDips/2:,numDips/2:] = m_LW_short*np.identity(numDips/2)
    all_masses = np.repeat(dipole_masses[np.newaxis, :, :], len(w_all), axis=0)

    dipole_damps = np.zeros( (len(w_all), np.int(numDips),np.int(numDips))) 
    dipole_damps[:,0,0] = gam_LW_long; dipole_damps[:,1,1] = gam_LW_long; dipole_damps[:,2,2] = gam_LW_long; dipole_damps[:,3,3] = gam_LW_long
    dipole_damps[:,4,4] = gam_LW_short; dipole_damps[:,5,5] = gam_LW_short; dipole_damps[:,6,6] = gam_LW_short; dipole_damps[:,7,7] = gam_LW_short
    all_damps = dipole_damps

    for i in range(0,np.int(numDips)):
        for j in range(0,np.int(numDips)):
            r_ij = np.array([ dip_origins[i,0]-dip_origins[j,0] , dip_origins[i,1]-dip_origins[j,1] ])
            mag_rij = scipy.linalg.norm(r_ij)
            if i == j:
                # print 'w_all = ', "%e" % w_all[0]
                # print 'all_damps = ', "%e" % all_damps[0,i,i]
                # print 'all_wsps = ', "%e" % all_wsps[0,i,i]
                # print 'all_masses = ', "%e" % all_masses[0,i,i]
                H[:,i,j] = (-w_all**2 - 1j*w_all*all_damps[:,i,i] + all_wsps[:,i,i]**2)*all_masses[:,i,i]
                # print i, j, H[:,i,j].shape
            if mag_rij == 0 and i != j: # to prevent divide by zero errors
                H[:,i,j] = 0.0
            if mag_rij != 0: 
                H[:,i,j] = -make_g(w_all=w_all, i=i, j=j, m=all_masses[:,i,i])
    invert_H = np.linalg.inv(H)
    return invert_H

#make_invH(w_all=loadData()[0],wsp_long=1.3, wsp_short=1.6, gamNR_long=.1, gamNR_short=.1)


def make_alpha( #total alpha = e*x_tot*E_tot^-1
    w_all, # the range of wavelengths the Gam EELS is taken over, i.e. we need a val of GamEEL for many different wavelenths [1/s]
    wsp_long, 
    wsp_short,
    gamNR_long, 
    gamNR_short,
    cs,
    a,
    ):
    inverted_H = make_invH(w_all=w_all,wsp_long=wsp_long, wsp_short=wsp_short,gamNR_long=gamNR_long, gamNR_short=gamNR_short,cs=cs, a=a)
    alpha_tot = e**2*inverted_H
    return alpha_tot

#print make_alpha(w_all=loadData()[0],wsp_long=1.3, wsp_short=1.6, gamNR_long=.1, gamNR_short=.1)

def gammaEELS(
    w_all, # the range of wavelengths the Gam EELS is taken over, i.e. we need a val of GamEEL for many different wavelenths [1/s]
    wsp_long,
    wsp_short,
    gamNR_long,
    gamNR_short,
    ebeam_loc,
    cs,
    a):     
    v = 0.48 
    gamL = 1/np.sqrt(1-v**2)
    Gam_EELS_tot = 0
    alpha_tot = make_alpha(w_all=w_all, wsp_long=wsp_long, wsp_short=wsp_short,gamNR_long=gamNR_long,gamNR_short=gamNR_short,cs=cs, a=a)
    # for i in range(0,numDips): ## add up gam = const*imag(alph1 + alph2 + alph3 + ...)
    #     dip_loc = dip_origins[i,:]
    #     magR = np.linalg.norm(ebeam_loc-dip_loc)
    #     constants = 4.0*e**2/((hbar_eVs)*hbar_cgs*np.pi*(v*c)**4*gamL**2)*w_all**2*(kn(1,w_all*magR/(v*c*gamL)))**2
    #     alpha_i_dressed = np.sum(alpha_tot[:,i,:], axis=-1)
    #     Gam_EELS_i = constants*np.imag(alpha_i_dressed)
    #     Gam_EELS_tot = Gam_EELS_i + Gam_EELS_tot
    if ebeam_loc[0] == 0: #long axis#
        for i in range(1,4,2): ## add up gam = const*imag(alph1 + alph3)
            dip_loc = dip_origins[i,:]
            magR = np.linalg.norm(ebeam_loc-dip_loc)
            constants = 4.0*e**2/((hbar_eVs)*hbar_cgs*np.pi*(v*c)**4*gamL**2)*w_all**2*(kn(1,w_all*magR/(v*c*gamL)))**2
            alpha_i_dressed = np.sum(alpha_tot[:,i,:], axis=-1)
            Gam_EELS_i = constants*np.imag(alpha_i_dressed)
            Gam_EELS_tot = Gam_EELS_i + Gam_EELS_tot

    if ebeam_loc[0] ==-120E-7: # short axis#
        for i in range(0,2,1): ## add up gam = const*imag(alph0 + alph1)
            dip_loc = dip_origins[i,:]
            magR = np.linalg.norm(ebeam_loc-dip_loc)
            constants = 4.0*e**2/((hbar_eVs)*hbar_cgs*np.pi*(v*c)**4*gamL**2)*w_all**2*(kn(1,w_all*magR/(v*c*gamL)))**2
            alpha_i_dressed = np.sum(alpha_tot[:,i,:], axis=-1)
            Gam_EELS_i = constants*np.imag(alpha_i_dressed)
            Gam_EELS_tot = Gam_EELS_i + Gam_EELS_tot

    return Gam_EELS_tot #units of 1/eV



# gammaEELS(w_all=loadData()[0],wsp_long=1.2, wsp_short=1.3, gamNR_long=.1, gamNR_short=.1,ebeam_loc=np.array([0,358E-7]),cs=100E-7,a=20E-7)

def gammaEELS_fit(raw_w, wsp_long, wsp_short, gamNR_long, gamNR_short):
    raw_w = raw_w#[:len(raw_w)/2]

    gammaEEL_long = gammaEELS(w_all=raw_w,wsp_long=wsp_long, wsp_short=wsp_short, gamNR_long=gamNR_long*gaml_scale, gamNR_short=gamNR_short*gams_scale,ebeam_loc=np.array([0,358E-7]),cs=100E-7,a=40E-7)
    gammaEEL_short = gammaEELS(w_all=raw_w,wsp_long=wsp_long, wsp_short=wsp_short, gamNR_long=gamNR_long*gaml_scale, gamNR_short=gamNR_short*gams_scale,ebeam_loc=np.array([-120E-7, 192.42683E-7]),cs=100E-7,a=40E-7)
    fitit = gammaEEL_long/max(gammaEEL_long)

    #fitit = np.concatenate((gammaEEL_long/max(gammaEEL_long), gammaEEL_short/max(gammaEEL_short)), axis=0)
    return fitit


w_raw_concat = loadData()[0] #1/s 

gaml_scale = 0.2
gams_scale = 0.055


#cs_scale = 70 #(half length), it's allowed to range from 140-196 nm 
#a_scale = 20 #(half length), it's allowed to range from 40-100 nm
#~#~#~# Fitting Time #~#~#~#
        #w_l w_s gam_l gam_s
lower = [1.2, 1.4,  1,   1]
upper = [1.5, 2.,  10, 10]
    

params, params_covariance = optimize.curve_fit(gammaEELS_fit, loadData()[0], loadData()[1], bounds=[lower,upper])
print params

# print 'cs = ', np.round(params[4]*2*cs_scale)
# print 'a = ', np.round(params[5]*2*a_scale)

plt.subplot(2,1,1)

#func = gammaEELS_fit(raw_w=w_raw_concat,wsp_long=1.3, wsp_short=1.9, gamNR_long=.1, gamNR_short=.1)
# plt.plot(w_raw_concat[:len(w_raw_concat)/2]*hbar_eVs,  gammaEELS_fit(w_raw_concat,*params)[:len(w_raw_concat)/2],'k',label='Fit',linewidth=3)
# plt.plot(w_raw_concat[:len(w_raw_concat)/2]*hbar_eVs, loadData()[1][:len(loadData()[0])/2], color='dodgerblue', linestyle=':',label='Raw data',linewidth=3)

plt.plot(w_raw_concat*hbar_eVs,  gammaEELS_fit(w_raw_concat,*params),'k',label='Fit',linewidth=3)
plt.plot(w_raw_concat*hbar_eVs, loadData()[1], color='dodgerblue', linestyle=':',label='Raw data',linewidth=3)


# plt.subplot(2,1,2)
# plt.plot(w_raw_concat[len(w_raw_concat)/2:]*hbar_eVs, loadData()[1][len(w_raw_concat)/2:], color='green', linestyle=':',label='Raw data',linewidth=3)
# plt.plot(w_raw_concat[len(w_raw_concat)/2:]*hbar_eVs,  gammaEELS_fit(w_raw_concat,*params)[len(w_raw_concat)/2:],'k',label='Fit',linewidth=3)


# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel('Energy [eV]',fontsize=16)
# plt.ylabel('Noramlized EELS [a.u.]',fontsize=16)
# plt.title('Long axis dipole', fontsize=16)
# plt.legend(fontsize=14)

plt.show()