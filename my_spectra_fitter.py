import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import yaml

open_param_file = open('parameters.yaml')
param = yaml.load(open_param_file)
c = param['constants']['c']
hbar_eVs = param['constants']['hbar_eVs']
e = param['constants']['e']
eps_b = np.sqrt(param['n_b'])

n = eps_b**2
nm_to_per_s = 2*np.pi*c/(n)*1e7 # omega = this variable / lambda (in nm)
um_to_per_s = 2*np.pi*c/(n)*1e4 # omega = this variable / lambda (in um)
dim=2
observable = 'gammaEELS'

load_coordinates = param['load_coords']
coordinates = np.loadtxt(load_coordinates)
x_coords = coordinates[:,0]
numPart = np.float(len(x_coords))
dip_coords = np.zeros((np.int(numPart),np.int(dim)),dtype=np.double)

for row in range(0,np.int(numPart)):
    dip_coords[row,:] = np.array((coordinates[row,0], coordinates[row,1]))
dip_coords = np.append(dip_coords, dip_coords,axis=0)*10**(-7)
y_coords = dip_coords[0:np.int(numPart),0]*10**7
z_coords = dip_coords[0:np.int(numPart),1]*10**7


def loadData():
	dir = param['spectra_fitting']['which_dipole']
	path = param['spectra_fitting']['path']
	data = np.loadtxt(path,dtype=float)
	observable = param['spectra_fitting']['observable']
	code = param['spectra_fitting']['which_code']
	if code == 'bem':
		w = data[:,0] 
		r = 100.e-7
		area_cross = np.pi*r**2
		ext_effic = data[:,1]/( (10**7)**2 * area_cross) # last term converts nm^2 to cm^2 and then to unitless
		abs_effic = data[:,2]/( (10**7)**2 * area_cross) # last term converts nm^2 to cm^2 and then to unitless
		sca_effic = data[:,3]/( (10**7)**2 * area_cross)# last term converts nm^2 to cm^2 and then to unitless
		if observable == 'ext': effic = ext_effic
		if observable == 'abs': effic = abs_effic
		if observable == 'sca': effic = sca_effic

	if code == 'dda':
		w = 2*np.pi*c / (n*data[:,1]*10**-4)*hbar_eVs # frequency in eVs
		area_cross = np.pi*data[0,0]**2 * 1e-8 #convert um**2 to cm**2
		ext_effic = data[:,2]
		abs_effic = data[:,3]
		sca_effic = data[:,4]
		if observable == 'ext': effic = ext_effic
		if observable == 'abs': effic = abs_effic
		if observable == 'sca': effic = sca_effic

	allData = np.column_stack([w, effic])
	allData_sort = allData[allData[:,0].argsort()[::-1]]

	# if dir == 'bem':
	# 	idx = np.where(allData_sort[:,0] > 2.5)
	# 	allData_sort = np.delete(allData_sort, idx, axis=0)
	# 	idx = np.where(allData_sort[:,0] < 1.5)
	# 	allData_sort = np.delete(allData_sort, idx, axis=0)

	# if dir == 'trans':
	# 	idx = np.where(allData_sort[:,0] > 1.58)
	# 	allData_sort = np.delete(allData_sort, idx, axis=0)
	# 	idx = np.where(allData_sort[:,0] < 1.2)
	# 	allData_sort = np.delete(allData_sort, idx, axis=0)

	# if dir == 'long':
	# 	idx = np.where(allData_sort[:,0] > 1.5)		
	# 	allData_sort = np.delete(allData_sort, idx, axis=0)
	# 	idx = np.where(allData_sort[:,0] < 1.)	
	# 	allData_sort = np.delete(allData_sort, idx, axis=0)
	
	w = np.asarray(allData_sort[:,0]) 
	effic_sim = np.asarray(allData_sort[:,1]) 
	return [w, effic_sim, area_cross]

def eps_p(kind):
	if kind == 'JC_data':
		JC = np.loadtxt('auJC.dat')
		w = JC[:,0]
		return JC[:,1]

	if kind == 'drude':
		JC = np.loadtxt('auJC.dat')
		w = JC[:,0]
		#return #blah

def prolate(direction, c_s, a_s):
	e_s = np.sqrt( (c_s**2-a_s**2) /c_s**2)
	L_z = (1-e_s**2)/(e_s**3)*(-e_s+np.arctanh(e_s))
	L_y = (1.-L_z)/2.0 
	D_z = 3./4.*((1.0+e_s**2)/(1.0-e_s**2)*L_z+1)
	D_y = a_s/(2.*c_s)*(3./e_s*np.arctanh(e_s)-D_z)
	if direction == 'long': return L_z, D_z, c_s
	if direction == 'short': return L_y, D_y, a_s

def alpha(w,eps_p,a_s,c_s):
	V = 4.0/3.0*np.pi*a_s**2*c_s
	eps = eps_p / eps_b
	L, D, lE = prolate(direction,c_s,a_s)
	k = 2*np.pi*np.sqrt(eps_b)/w 
	alpha_r = V/(4*np.pi)*(eps-1)/(1+L*(eps-1))
	alpha = alpha_r/(1-k**2/lE*D*alpha_r-1j*2*k**3/3*alpha_r)

def abs(w,m_scaled,w0):
	area_cross = loadData()[2]
	scale = 10**-31 #
	m = m_scaled*scale
	gam_NR = 0.069
	gam_RAD = ( 2.*e**2 / (3.*m*c**3)*(w/hbar_eVs)**2 ) * hbar_eVs #in eV's, it takes w in units of eV as well (and internally converts)
 	return ( 4.*np.pi*w**2 * e**2 * gam_NR * hbar_eVs / (c * m * ((w0**2-w**2)**2 + ( (gam_NR+gam_RAD)*w)**2)) ) / area_cross #all in eV, or if not it's CGS

def ext(w,m_scaled,w0):
	area_cross = loadData()[2]
	scale = 10**-31 #long=31
	m = m_scaled*scale
	gam_NR = 0.069
	gam_RAD = ( 2.*e**2 / (3.*m*c**3)*(w/hbar_eVs)**2 ) * hbar_eVs #in eV's, it takes w in units of eV as well (and internally converts)
	gam_tot	= gam_NR + gam_RAD
	im_alpha = e**2/m*(w/hbar_eVs)*(gam_tot/hbar_eVs) / ( ( (w0/hbar_eVs)**2 - (w/hbar_eVs)**2)**2 + (w*gam_tot/hbar_eVs**2)**2 ) # let's switch everything back to cgs in this line
	return 4*np.pi*(w/hbar_eVs) / c * im_alpha / area_cross

def gammaEELS(w,m_scaled,w0, alpha):
    particle = 0
    v = 0.7*c
    gam_L = 1./(np.sqrt(1-v**2/c**2))
    R_part_beam = np.sqrt((y_coords[particle]-beam_spot[0])**2 + (z_coords[particle]-beam_spot[1])**2 )*10**-7
    return 4*e**2/(hbar_eVs**2*np.pi*v**4*gam_L**2)*e**2/m_scaled*w**2*special.kn(1, w/hbar_eVs*R_part_beam/(v*gam_L))**2*imag.alpha/(e**2/m_scaled)

if observable == 'ext':
 	params, params_covariance = optimize.curve_fit(ext, loadData()[0], loadData()[1],bounds=[0,np.inf])
	plt.plot(loadData()[0], ext(loadData()[0],*params),label='Fit')
	plt.title('Single Particle Exctinction Spectrum')

if observable == 'abs':
	params, params_covariance = optimize.curve_fit(abs, loadData()[0], loadData()[1])
	plt.plot(loadData()[0], abs(loadData()[0],*params),label='Fit')
	plt.title('Single Particle Absorption Spectrum')

plt.plot(loadData()[0], loadData()[1], label='Raw data')
plt.legend()
plt.show()

print params
loadData()

def writeFitParameters(params):
	file = open('fit_rod_long','w')
	file.write(str(params[0]*10**-31) + '\n')	#m (g)
	file.write(str(params[1]) + '\n') #w0 eV
	file.close()

writeFitParameters(params)
