import numpy as np
import matplotlib.pyplot as plt
from claires_tight import calculateEigen
from matplotlib.pyplot import figure
import yaml
open_param_file = open('parameters.yaml')
param = yaml.load(open_param_file)

eps_b = np.sqrt(param['n_b'])
c = param['constants']['c']
hbar_eVs = param['constants']['hbar_eVs']
e = param['constants']['e']
dim = param['constants']['dim']

prec = param['precision']
base_vector = param['dipole_directions']
load_coordinates = param['load_coords']
input_parameters = param['input_parameters']
numPart = 3

def seeModes():
    constants, ham_hist, total, dip_coords  = calculateEigen()
    #total = np.loadtxt('total.txt')
    #dip_coords = np.loadtxt('dip_coords.txt')
    w = total[:,0]
    base_vectors = np.loadtxt(base_vector)
    numPart = len(w)/dim
    v = total[:,1:np.int(numPart*dim)+1]
    y_coords = dip_coords[0:np.int(numPart),0]*10**7
    z_coords = dip_coords[0:np.int(numPart),1]*10**7
    unit_vector = np.zeros((np.int(numPart*dim),2),dtype=np.double)
    vector = np.zeros((np.int(numPart*dim),np.int(numPart)*4),dtype=np.double)
    vec_y_coord_all = np.zeros((np.int(numPart*dim),np.int(numPart*dim)),dtype=np.double)
    vec_z_coord_all = np.zeros((np.int(numPart*dim),np.int(numPart*dim)),dtype=np.double)

    for mode in range(0,np.int(numPart*dim)):
        magnitudes = np.array([v[mode]]).T
        maggys = np.column_stack([magnitudes, magnitudes])
        for particle in range(0,np.int(numPart*dim)):
            unit_vector[particle,:] = base_vectors[particle,:]/np.linalg.norm(base_vectors[particle,:])
        vector = maggys*unit_vector
        vec_y_coord_all[:,mode] = vector[:,0]
        vec_z_coord_all[:,mode] = vector[:,1]        
    fig = plt.figure(num=None, figsize=(11, 6), dpi=80, facecolor='w', edgecolor='k')   
    for mode in range(0,3):#np.int(numPart*dim)):
        #figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
        evec_ycoord = vec_y_coord_all[0:np.int(numPart),mode] + vec_y_coord_all[np.int(numPart):np.int(numPart*dim),mode]
        evec_zcoord = vec_z_coord_all[0:np.int(numPart),mode] + vec_z_coord_all[np.int(numPart):np.int(numPart*dim),mode]
        evalue = round(w[mode], prec)
        evalue_nm = hbar_eVs*c*2*np.pi*10**7/(evalue*eps_b**2)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.subplot(2,5,mode+1)
        #plt.title('Mode = %d;' % (mode) + ' eval = %.3f eV' % (evalue), fontsize=14)
        plt.title('%.0f nm' % (evalue_nm), fontsize=14)
        plt.xlim((min(y_coords)-100,max(y_coords)+100))
        plt.ylim((min(z_coords)-100,max(z_coords)+100))
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        #plt.xticks(fontsize=6)
        #plt.yticks(fontsize=6)
        mag_tot = []
        y = []
        z = []
        for particle in range(0, np.int(numPart)):
            mag = np.sqrt( (evec_ycoord[particle])**2 + (evec_zcoord[particle])**2)
            y = np.append(y_coords[particle], y)
            z = np.append(z_coords[particle], z)
            mag_tot = np.append(mag, mag_tot)
            abs_eff = mag_tot**2 
        #plt.scatter(y, z, c=abs_eff,s=100, cmap='viridis')
        plt.quiver(y_coords,z_coords,evec_ycoord,evec_zcoord, pivot='mid', scale=2.1, minshaft=5,headwidth=5)
        plt.axis('equal') 
    plt.suptitle('Full closed 1', fontsize=18)
    #plt.suptitle('X=50 = %d' % prec, fontsize=18)
    fig.savefig('normal_modes.pdf')

seeModes()

''' Plot Magnitudes'''
# for particle in range(0, np.int(numPart)):
#     mag = np.sqrt( (evec_ycoord[particle])**2 + (evec_zcoord[particle])**2)
#     #plt.text(y_coords[particle], z_coords[particle] - 20, "{:.1e}".format(mag), size=5, ha="center", va="center",bbox=dict(boxstyle="round",ec='w', fc='mediumaquamarine'))
#     #y = outlines[:,0]
#     #z = outlines[:,1]
#     #yp = np.cos(np.pi/2)*y - np.sin(np.pi/2)*z
#     #zp = np.sin(np.pi/2)*y + np.cos(np.pi/2)*z
#     #y = yp + y_coords[particle]
#     #z = zp + z_coords[particle]
#     y = np.append(y_coords[particle], y)
#     z = np.append(z_coords[particle], z)
#     mag_tot = np.append(mag, mag_tot)
#     m = constants[3]
#     gam_NR = constants[4]
#     #abs = 4*np.pi*(evalue/hbar_eVs)**2/c*(m/e**2)*gam_NR*mag_tot**2
#     cross_area = np.pi * (0.5*200.*10**-7)**2
#     abs_eff = mag_tot**2 
# plt.scatter(y, z, c=abs_eff,s=100, cmap='viridis')
# plt.colorbar()
# plt.axis('equal')


