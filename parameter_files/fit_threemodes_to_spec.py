import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sys
from decimal import Decimal
import math
import yaml
from scipy import optimize
from scipy.special import kn

open_param_file = open('parameters.yaml')
param = yaml.load(open_param_file)
c = param['constants']['c']
eps_b = 1.0

hbar_eVs = param['constants']['hbar_eVs']
e = param['constants']['e']
hbar_cgs = param['constants']['hbar_cgs']

prec = param['precision']
n=1
um_to_per_s = 2*np.pi*c/(n)*1e4 # omega = this variable / lambda (in um)
#inputdata = np.loadtxt('inputs_spheres.txt', skiprows=1)
inputs = np.loadtxt(str('../')+param['inputs'],skiprows=1)

def loadData(which):
    if which == 'Dl':
        #data_long = np.loadtxt('/Users/clairewest/werk/research/penrose/my_sims/monomer/Spectrum_2DS_long',skiprows=1)
        # w_long = data_long[:,1] #eV
        # effic_long = data_long[:,2]
        # allData_long = np.column_stack([w_long, effic_long])
        # allData_sortlong = allData_long[allData_long[:,0].argsort()[::-1]]
        # idx = np.where(allData_sortlong[:,0] > 1.9)     #1.9
        # allData_sortlong = np.delete(allData_sortlong, idx, axis=0)
        # idx = np.where(allData_sortlong[:,0] <= 1.1)  #1.1
        # allData_sortlong = np.delete(allData_sortlong, idx, axis=0)
        # w = np.asarray(allData_sortlong[:,0]) 
        # effic_sim = np.asarray(allData_sortlong[:,1])
        data_long = np.loadtxt('/Users/clairewest/werk/research/penrose/experiment/Files For Claire/Monomer_Subtracted.csv',delimiter=',',skiprows=1)
        w_long = data_long[:,0] #eV
        effic_long = data_long[:,1]
        allData_long = np.column_stack([w_long, effic_long])
        allData_sortlong = allData_long[allData_long[:,0].argsort()[::-1]]
        idx = np.where(allData_sortlong[:,0] > 1.3)     #1.9
        allData_sortlong = np.delete(allData_sortlong, idx, axis=0)
        idx = np.where(allData_sortlong[:,0] <= .3)  #1.1
        allData_sortlong = np.delete(allData_sortlong, idx, axis=0)
        w = np.asarray(allData_sortlong[:,0]) 
        effic_sim = np.asarray(allData_sortlong[:,1])



    
    if which == 'Ds':
        data_short = np.loadtxt('/Users/clairewest/werk/research/penrose/my_sims/monomer/Spectrum_2DS_short',skiprows=1)
        w_short = data_short[:,1] # eV
        effic_short = data_short[:,2]
        allData_short = np.column_stack([w_short, effic_short])
        allData_sortshort = allData_short[allData_short[:,0].argsort()[::-1]]
        idx = np.where(allData_sortshort[:,0] >= 1.9)     
        allData_sortshort = np.delete(allData_sortshort, idx, axis=0)
        idx = np.where(allData_sortshort[:,0] <= .5)  
        allData_sortshort = np.delete(allData_sortshort, idx, axis=0)

        w = np.asarray(allData_sortshort[:,0]) 
        effic_sim = np.asarray(allData_sortshort[:,1])   

        # data_long = np.loadtxt('/Users/clairewest/werk/research/penrose/experiment/Files For Claire/Monomer_Subtracted.csv',delimiter=',',skiprows=1)
        # w_long = data_long[:,0] #eV
        # effic_long = data_long[:,2]
        # allData_long = np.column_stack([w_long, effic_long])
        # allData_sortlong = allData_long[allData_long[:,0].argsort()[::-1]]
        # idx = np.where(allData_sortlong[:,0] > 1.3)     #1.9
        # allData_sortlong = np.delete(allData_sortlong, idx, axis=0)
        # idx = np.where(allData_sortlong[:,0] <= .3)  #1.1
        # allData_sortlong = np.delete(allData_sortlong, idx, axis=0)
        # w = np.asarray(allData_sortlong[:,0]) 
        # effic_sim = np.asarray(allData_sortlong[:,1])

    return [w, effic_sim/max(effic_sim)]



# gRscale = .01
# amp_scale = 45.
def gammaEELS(
    w_all, # the range of wavelengths the Gam EELS is taken over, i.e. we need a val of GamEEL for many different wavelenths [1/s]
    w0,
    gamR,
    ebeam_loc,
    amp,
    ):     
    v = 0.48 
    gamL = 1/np.sqrt(1-v**2)
    m = (2.0*e**2)/(3.0*gamR*hbar_eVs*c**3)
    gam = 0.07 + gamR*w_all**2
    alpha = e**2/m * hbar_eVs**2/(-w_all**2 - 1j*gam*w_all + w0**2)
    rhomb_loc = np.array([0,192])*1E-7 # row doesn't matter, we're just fitting to the center of rhomb
    magR = np.linalg.norm(ebeam_loc-rhomb_loc)
    constants = 4.0*e**2/((hbar_eVs)*hbar_cgs*np.pi*(v*c)**4*gamL**2)*(w_all/hbar_eVs)**2*(kn(1,(w_all/hbar_eVs)*magR/(v*c*gamL)))**2
   # print constants
    Gam_EELS = constants*np.imag(alpha)
    return amp*Gam_EELS #units of 1/eV

# #plt.plot(loadData(which='Dl')[0], gammaEELS(w_all=loadData(which='Dl')[0], w0=1.3, gamNR=0.01, gamR=.1, ebeam_loc=np.array([0,358E-7])))
# #plt.show()

# def fit_gammaEELSlong(raw_w, w0_fit, gamR_fit, amp):
#     ebeam_loc=np.array([0,358E-7])
#     gammaEEL_long = gammaEELS(w_all=raw_w, w0=w0_fit, gamR=gamR_fit*gRscale,amp=amp*amp_scale,ebeam_loc=ebeam_loc)
#     return gammaEEL_long

# def plot_n_fit_long():
#     #       w_l, gam_l, m_l
#     lower = [1.25, 1,  1]
#     upper = [2., 5.,  5]
#     guess = [1.3, 1, 1]

#     w_rawdata = loadData(which='Dl')[0]
#     eel_rawdata = loadData(which='Dl')[1]
#     params, params_covariance = optimize.curve_fit(fit_gammaEELSlong, w_rawdata, eel_rawdata, bounds=[lower,upper],p0=guess)
#     print params
#     print 'w_l = ', '%.2f' % params[0]
#     print 'gamR = ', '%.2f' %  (params[1]*gRscale)
#     print 'm = ', '%.3e' % ((2.0*e**2)/(3.0*(params[1]*gRscale)*hbar_eVs*c**3))
#     plt.subplot(1,1,1)
#     #plt.plot(w_rawdata*hbar_eVs,  gammaEELS(w_all=w_rawdata,w0=1.32/hbar_eVs, gamNR=0.1/hbar_eVs, m=1E-27, ebeam_loc=np.array([0,358E-7])),'k',label='Fit',linewidth=3)
#     #plt.plot(w_rawdata,  fit_gammaEELSlong(w_rawdata, guess[0], guess[1]*gRscale, guess[2]*amp_scale),'r',label='Fit',linewidth=3)
#     plt.plot(w_rawdata,  fit_gammaEELSlong(w_rawdata,*params),'k',label='Fit',linewidth=3)
#     plt.plot(w_rawdata, eel_rawdata, color='dodgerblue', linestyle=':',label='Raw data',linewidth=3)
#     plt.show()
# plot_n_fit_long()

################################################
################################################

# gaml_scale = .08; gamq_scale = 0.02
# ampl_scale = 300; ampq_scale = 5.E3

gaml_scale = .1; gamq_scale = 0.04
ampl_scale = 10; ampq_scale = 80

def fit_gammaEELSlongnquad(raw_w, w0l, w0q, gamRl, gamRq, ampl, ampq):
    ebeam_loc = np.array([0,358E-7])
    gammaEEL_long = gammaEELS(w_all=raw_w,w0=w0l, gamR=gamRl*gaml_scale,amp=ampl*ampl_scale,ebeam_loc=ebeam_loc)
    gammaEEL_quad = gammaEELS(w_all=raw_w,w0=w0q, gamR=gamRq*gamq_scale,amp=ampq*ampq_scale,ebeam_loc=ebeam_loc)
    return gammaEEL_long + gammaEEL_quad

def plot_n_fit_longnquad():
    #       w_s, w_q, g_s, g_q, m_s, m_q
    lower = [.4, 1.,  1,   1.,   1,    1]
    upper = [2.0, 1.5, 10,   10, 10.,   10]
    guess = [.7, 1.1,  5, 1.35,   5,    5]

    w_rawdata = loadData(which='Dl')[0]
    eel_rawdata = loadData(which='Dl')[1]
    params, params_covariance = optimize.curve_fit(fit_gammaEELSlongnquad, w_rawdata, eel_rawdata, bounds=[lower,upper],p0=guess)
    print params
    print 'w_l = ', '%.2f' % params[0]
    print 'w_q = ', '%.2f' % params[1]
    print 'gam_l = ','%.2f' %  (params[2]*gaml_scale)
    print 'gam_q = ','%.2f' % (params[3]*gamq_scale)
    print 'm_l = ', '%.3e' % ((2.0*e**2)/(3.0*(params[2]*gaml_scale)*hbar_eVs*c**3))
    print 'm_q = ', '%.3e' % ((2.0*e**2)/(3.0*(params[3]*gamq_scale)*hbar_eVs*c**3))

    plt.subplot(1,1,1)
    plt.plot(w_rawdata,  fit_gammaEELSlongnquad(w_rawdata,*params),'k',label='Fit',linewidth=3)
    plt.plot(w_rawdata, eel_rawdata, color='dodgerblue', linestyle=':',label='Raw data',linewidth=3)
    plt.show()

#plot_n_fit_longnquad()

################################################
################################################

# gams_scale = .15; gamq_scale = 0.02
# amps_scale = 1.E2; ampq_scale = 3.7E2

gams_scale = .02; gamq_scale = 0.02
amps_scale = 50; ampq_scale = 80

def fit_gammaEELSshortnquad(raw_w, w0s, w0q, gamRs, gamRq, amps, ampq):
    ebeam_loc = np.array([-120E-7, 192.42683E-7])
    gammaEEL_short = gammaEELS(w_all=raw_w,w0=w0s, gamR=gamRs*gams_scale,amp=amps*amps_scale,ebeam_loc=ebeam_loc)
    gammaEEL_quad = gammaEELS(w_all=raw_w,w0=w0q, gamR=gamRq*gamq_scale,amp=ampq*ampq_scale,ebeam_loc=ebeam_loc)
    return gammaEEL_short + gammaEEL_quad

def plot_n_fit_shortnquad():
    #       w_s, w_q, g_s, g_q, m_s, m_q
    lower = [1.4, 1.4,  1, 1., 1,    1]
    upper = [2., 2.5, 10, 10, 10., 10]

    guess = [1.6, 1.8,  5, 1.35, 5,   5]

    w_rawdata = loadData(which='Ds')[0]
    eel_rawdata = loadData(which='Ds')[1]
    params, params_covariance = optimize.curve_fit(fit_gammaEELSshortnquad, w_rawdata, eel_rawdata, bounds=[lower,upper],p0=guess)
    print params
    print 'w_s = ', '%.2f' % params[0]
    print 'w_q = ', '%.2f' % params[1]
    print 'gam_s = ','%.2f' %  (params[2]*gams_scale)
    print 'gam_q = ','%.2f' % (params[3]*gamq_scale)
    print 'm_s = ', '%.3e' % ((2.0*e**2)/(3.0*(params[4]*gams_scale)*hbar_eVs*c**3))
    print 'm_q = ', '%.3e' % ((2.0*e**2)/(3.0*(params[5]*gamq_scale)*hbar_eVs*c**3))

    plt.subplot(1,1,1)
    plt.plot(w_rawdata,  fit_gammaEELSshortnquad(w_rawdata,*params),'k',label='Fit',linewidth=3)
    plt.plot(w_rawdata, eel_rawdata, color='dodgerblue', linestyle=':',label='Raw data',linewidth=3)

    plt.show()


plot_n_fit_shortnquad()
