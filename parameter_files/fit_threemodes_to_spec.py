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
hbar_cgs = param['constants']['hbar_cgs']

prec = param['precision']
n=1
um_to_per_s = 2*np.pi*c/(n)*1e4 # omega = this variable / lambda (in um)
inputdata = np.loadtxt('inputs_spheres.txt', skiprows=1)

def loadData(which):
    if which == 'Dl':
        data_long = np.loadtxt('/Users/clairewest/werk/research/penrose/my_sims/monomer/Spectrum_2DS_long',skiprows=1)
        w_long = um_to_per_s / data_long[:,0] # needs to be in 1/s
        effic_long = data_long[:,2]
        allData_long = np.column_stack([w_long, effic_long])
        allData_sortlong = allData_long[allData_long[:,0].argsort()[::-1]]
        idx = np.where(allData_sortlong[:,0] >= 1.6/hbar_eVs)     
        allData_sortlong = np.delete(allData_sortlong, idx, axis=0)
        idx = np.where(allData_sortlong[:,0] <= 1./hbar_eVs)  
        allData_sortlong = np.delete(allData_sortlong, idx, axis=0)
        w = np.asarray(allData_sortlong[:,0]) 
        effic_sim = np.asarray(allData_sortlong[:,1])
    
    if which == 'Ds':
        data_short = np.loadtxt('/Users/clairewest/werk/research/penrose/my_sims/monomer/Spectrum_2DS_short',skiprows=1)
        w_short = um_to_per_s / data_short[:,0] # needs to be in 1/s
        effic_short = data_short[:,2]
        allData_short = np.column_stack([w_short, effic_short])
        allData_sortshort = allData_short[allData_short[:,0].argsort()[::-1]]
        idx = np.where(allData_sortshort[:,0] >= 2./hbar_eVs)     
        allData_sortshort = np.delete(allData_sortshort, idx, axis=0)
        idx = np.where(allData_sortshort[:,0] <= .5/hbar_eVs)  
        allData_sortshort = np.delete(allData_sortshort, idx, axis=0)

        w = np.asarray(allData_sortshort[:,0]) 
        effic_sim = np.asarray(allData_sortshort[:,1])   

    return [w, effic_sim]

def gammaEELS(
    w_all, # the range of wavelengths the Gam EELS is taken over, i.e. we need a val of GamEEL for many different wavelenths [1/s]
    w0,
    gamNR,
    m,
    ebeam_loc,
    ):     
    v = 0.48 
    gamL = 1/np.sqrt(1-v**2)
    gam = gamNR + w_all**2*(2.0*e**2)/(3.0*m*c**3)

    alpha = e**2/m * 1/(-w_all**2-1j*gam*w_all+w0**2)
    dip_loc = inputdata[0,0:2]*1E-7 #row doesn't matter, we're just fitting to the center of rhomb
    magR = np.linalg.norm(ebeam_loc-dip_loc)
    constants = 4.0*e**2/((hbar_eVs)*hbar_cgs*np.pi*(v*c)**4*gamL**2)*w_all**2#*(kn(1,w_all*magR/(v*c*gamL)))**2
    Gam_EELS = constants*np.imag(alpha)
    return Gam_EELS #units of 1/eV

gam_scale = 0.1
mscale = 1E-34

def fit_gammaEELSlong(raw_w, w0, gamNR, m):

    ebeam_loc=np.array([0,358E-7])
    gammaEEL_long = gammaEELS(w_all=raw_w,w0=w0/hbar_eVs, gamNR=gamNR*gam_scale/hbar_eVs, m=m*mscale,ebeam_loc=ebeam_loc)
    return gammaEEL_long

def plot_n_fit_long():
    #       w_l, gam_l, m_l
    lower = [1.2, 1,  1]
    upper = [1.5, 10.,  10]
    guess = [1.35, 5, 5]

    w_rawdata = loadData(which='Dl')[0]
    eel_rawdata = loadData(which='Dl')[1]
    params, params_covariance = optimize.curve_fit(fit_gammaEELSlong, w_rawdata, eel_rawdata, bounds=[lower,upper],p0=guess)
    print params
    print 'w_l = ', '%.2f' % params[0]
    print 'gam_l = ', '%.2f' %  (params[1]*gam_scale)
    print 'm_l = ', '%.2e' % (params[2]*mscale)

    plt.subplot(1,1,1)
    plt.plot(w_rawdata*hbar_eVs,  fit_gammaEELSlong(w_rawdata,*params),'k',label='Fit',linewidth=3)
    plt.plot(w_rawdata*hbar_eVs, eel_rawdata, color='dodgerblue', linestyle=':',label='Raw data',linewidth=3)
    plt.show()


gams_scale = .1; gamq_scale = 0.1
ms_scale = 1E-34; mq_scale = 2E-34

def fit_gammaEELSshortnquad(raw_w, w0s, w0q, gamNRs, gamNRq, ms, mq):
    ebeam_loc = np.array([-120E-7, 192.42683E-7])
    gammaEEL_short = gammaEELS(w_all=raw_w,w0=w0s/hbar_eVs, gamNR=gamNRs*gams_scale/hbar_eVs, m=ms*ms_scale,ebeam_loc=ebeam_loc)
    gammaEEL_quad = gammaEELS(w_all=raw_w,w0=w0q/hbar_eVs, gamNR=gamNRq*gamq_scale/hbar_eVs, m=mq*mq_scale,ebeam_loc=ebeam_loc)
    return gammaEEL_short + gammaEEL_quad

def plot_n_fit_shortnquad():
    #       w_s, w_q, g_s, g_q, m_s, m_q
    lower = [1.4, 1.4,  1, 1., 1,    1]
    upper = [1.8, 2.0, 10, 10, 10., 10]

    guess = [1.6, 1.8,  5, 1.35, 5,   5]

    w_rawdata = loadData(which='Ds')[0]
    eel_rawdata = loadData(which='Ds')[1]
    params, params_covariance = optimize.curve_fit(fit_gammaEELSshortnquad, w_rawdata, eel_rawdata, bounds=[lower,upper],p0=guess)
    print params
    print 'w_s = ', '%.2f' % params[0]
    print 'w_q = ', '%.2f' % params[1]
    print 'gam_s = ','%.2f' %  (params[2]*gams_scale)
    print 'gam_q = ','%.2f' % (params[3]*gamq_scale)
    print 'm_s = ', '%.2e' % (params[4]*ms_scale)
    print 'm_q = ', '%.2e' % (params[5]*mq_scale)

    plt.subplot(1,1,1)
    plt.plot(w_rawdata*hbar_eVs,  fit_gammaEELSshortnquad(w_rawdata,*params),'k',label='Fit',linewidth=3)
    plt.plot(w_rawdata*hbar_eVs, eel_rawdata, color='dodgerblue', linestyle=':',label='Raw data',linewidth=3)
    plt.show()

#plot_n_fit_shortnquad()
plot_n_fit_long()

