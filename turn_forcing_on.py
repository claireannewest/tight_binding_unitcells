import numpy as np
import matplotlib.pyplot as plt
from claires_tight import calculateEigen
from scipy import special
from matplotlib.pyplot import figure
from matplotlib import cm
import yaml
open_param_file = open('parameters.yaml')
param = yaml.load(open_param_file)

eps_b = np.sqrt(param['n_b'])
c = param['constants']['c']
hbar_eVs = param['constants']['hbar_eVs']
e = param['constants']['e']
dim = param['constants']['dim']
force = param['turn_forcing']['force']
prec = param['precision']

outlines = 'no'

def forcing():
  coordinates = np.loadtxt(param['load_coords'])
  constants, final_hams, total, dip_coords  = calculateEigen() 
  w = total[:,0]
  [eps_b, c, hbar_eVs, m_y, m_z, gam_NR] = constants
  numPart = len(dip_coords[:,0])/dim
  y_coords = dip_coords[0:np.int(numPart),0]*10**7
  z_coords = dip_coords[0:np.int(numPart),1]*10**7

  for mode in range(0,np.int(numPart*dim)):
    figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
    H = final_hams[:, np.int(numPart*dim*mode) : np.int(numPart*dim*(mode+1))]
    omega = w[mode]
    A = H - np.identity(np.int(numPart*dim))*omega**2
    I0 = 1.e12 #erg/s/cm^2, in SI it's 10^9 W/m^2
    E0 = np.sqrt(I0*8.0*np.pi/(c*eps_b**2))
    b=[]
    E_y = []
    F_y = []
    if force == 'y_plane':
      F_y = np.ones(np.int(numPart))*e*E0/m_y*hbar_eVs**2
      F_z = np.ones(np.int(numPart))*0
      b = np.append(F_y, F_z)

    if force == 'z_plane':
      F_y = np.ones(np.int(numPart))*0
      F_z = np.ones(np.int(numPart))*e*E0/m_z*hbar_eVs**2
      b = np.append(F_y, F_z)

    if force == 'e_beam':
      y_offset = 2
      z_offset = 0
      beam_particle = 0
      beam_spot = np.array( [y_coords[beam_particle]+y_offset, z_coords[beam_particle]+z_offset])
      E_y = np.ones(np.int(numPart))*0 ##initialize
      E_z = np.ones(np.int(numPart))*0 ##initialize

      for particle in range(0, np.int(numPart)):
        v = 0.7*c
        gam_L = 1./(np.sqrt(1-v**2/c**2))
        R_part_beam = np.sqrt((y_coords[particle]-beam_spot[0])**2 + (z_coords[particle]-beam_spot[1])**2 )*10**-7
        E_el= -2.*e*omega/hbar_eVs/(v**2*gam_L**2)*gam_L*special.kn(1, omega/hbar_eVs*R_part_beam/(v*gam_L))
        # if np.round(beam_spot[0]) == 0:
        #   phi = np.pi/2.0
        #else:
        phi = np.arctan( (z_coords[particle]-beam_spot[1]) / (y_coords[particle]-beam_spot[0]) )
        E_el_y = E_el*np.cos(phi)
        E_el_z = E_el*np.sin(phi)
        E_y[particle] = E_el_y  
        E_z[particle] = E_el_z
      F_y = -E_y*e/m_y*hbar_eVs**2
      F_z = -E_z*e/m_z*hbar_eVs**2
      b = np.append(F_y, F_z)

    print b
    vec = np.linalg.solve(A,b)
    #print np.allclose(np.dot(A, vec), b, atol=10**(-prec))
    evalue = round(w[mode], prec)
    vec_ycoord = vec[0:np.int(numPart)]*10**7
    vec_zcoord = vec[np.int(numPart):np.int(numPart*dim)]*10**7
    mag_tot = []
    y = []
    z = []
    for particle in range(0, np.int(numPart)):
      mag = np.sqrt(vec_ycoord[particle]**2 + vec_zcoord[particle]**2)
      plt.text(y_coords[particle], z_coords[particle] - 10, "{:.1e}".format(mag), size=15, ha="center", va="center",bbox=dict(boxstyle="round",ec='w', fc='mediumaquamarine'))
      if outlines != 'no': 
        outline_data = np.loadtxt(outlines) #oriented vertically
        y = outline_data[:,0]
        z = outline_data[:,1]
        yp = np.cos(np.pi/2)*y - np.sin(np.pi/2)*z
        zp = np.sin(np.pi/2)*y + np.cos(np.pi/2)*z
        y = yp + y_coords[particle]
        z = zp + z_coords[particle]
        plt.scatter(y, z,color='mediumslateblue')

      y = np.append(y_coords[particle], y)
      z = np.append(z_coords[particle], z)
      mag_tot = np.append(mag, mag_tot)
      m = constants[3]
      gam_NR = constants[4]
      abs = 4*np.pi*(evalue/hbar_eVs)**2/c*(m/e**2)*gam_NR*mag_tot
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title('Mode = %d;' % (mode) + ' eval = %.3f eV' % (evalue), fontsize=20)

    plt.xlim((min(y_coords)-100,max(y_coords)+100))
    plt.ylim((min(z_coords)-100,max(z_coords)+100))

    if min(np.round(mag_tot)) == max(np.round(mag_tot)):
      print 'enter', mode
      viridis = cm.get_cmap('viridis', 12)
      plt.scatter(y, z, c=mag_tot/mag_tot,s=1000, vmin=0, vmax=max(mag_tot))
      cbar = plt.colorbar(format='%.0e',ticks=[0, max(mag_tot)])
      cbar.ax.tick_params(labelsize=14) 

    else:
      plt.scatter(y, z, c=mag_tot, s=1000, cmap='viridis',vmin=min(mag_tot), vmax=max(mag_tot))
      cbar = plt.colorbar(format='%.0e',ticks=[min(mag_tot), max(mag_tot)])
      cbar.ax.tick_params(labelsize=14) 

    if force == 'e_beam':
      plt.scatter( y_coords[beam_particle]+y_offset*20, z_coords[beam_particle]+z_offset*20 ,marker='+',c='k')

    plt.quiver(y_coords,z_coords,vec_ycoord,vec_zcoord)#,  pivot='mid',scale=1.3*10**4,minshaft=5,headwidth=6)
    #plt.show()
    #title = 'mode_' + str(mode) + ' eval_' + str(evalue)
    #np.savetxt(title, np.c_[y, z, y_coords, z_coords,vec_ycoord,vec_zcoord])

forcing()


