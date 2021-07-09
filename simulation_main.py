# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
Copyright (c) 2021 Groeblacher Lab

Author(s): Bas Hensen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------------------------------------------------------------------------------

Main file to reproduce the simulation figures Fig. 2d, Fig. 3, Fig. S3, Fig. S6 of:

Optomechanical quantum teleportation
Niccol`o Fiaschi,1,∗ Bas Hensen,1,∗ Andreas Wallucks,1 Rodrigo
Benevides,1,2 Jie Li,1,3 Thiago P. Mayer Alegre,2 and Simon Groeblacher1,

1 Kavli Institute of Nanoscience, Department of Quantum Nanoscience,
Delft University of Technology, 2628CJ Delft, The Netherlands
2 Photonics Research Center, Applied Physics Department,
Gleb Wataghin Physics Institute, P.O. Box 6165,
University of Campinas – UNICAMP, 13083-970 Campinas, SP, Brazil
3 Zhejiang Province Key Laboratory of Quantum Technology and Device,
Department of Physics, Zhejiang University, Hangzhou 310027, China

arXiv:2104.02080 [quant-ph]

(Dated: July 06, 2021)

"""

import logging
import matplotlib.pyplot as plt
import numpy as np

# local imports from cwd
from teleportation_simulation import teleportation_demonstration
from entanglement_simulation import entanglement_simulation

n_th_0_A = 0.049 # ref. Fig.S2b
zeta_inst_A = 0.91 # ref. Fig.S2b
n_th_0_B = 0.032 # ref. Fig.S2b
zeta_inst_B = 0.88 # ref. Fig.S2b

zeta_delayed_A = 3.4 # ref. Fig.S2c
zeta_delayed_B = 4.1 # ref. Fig.S2c

gcc_A = 4.81 # ref. Fig.2c
gcc_B = 9.76 # ref. Fig.2c

scattering_blue = 0.012 # ref. Main text
scattering_red  = 0.026 # ref. Main text

def get_simulation_parameters(): # These are the simulation paramters used in all figures, unless otherwise noted below
    return dict(

                scattering_blue = scattering_blue,             	# blue pulse scattering probability for each device 
                scattering_red = scattering_red,           		# red readout pulse scattering probability for each device
                

                n_th_blue_H = n_th_0_A + zeta_inst_A*scattering_blue,   # initial thermal population of mechanical mode for device in arm H
                n_th_blue_V = n_th_0_B + zeta_inst_B*scattering_blue,   # initial thermal population of mechanical mode for device in arm V
                                                                    	# from assymetry measurement versus power, 
                                                        		        # see SI section 2 and Fig S2b for details


                n_th_red_H = (1/(gcc_A-1)-scattering_blue) \
                              - (n_th_0_A + zeta_inst_A*scattering_blue),	# added thermal population after red readout pulse for device in arm H
                n_th_red_V = (1/(gcc_B-1)-scattering_blue) \
                              - (n_th_0_B + zeta_inst_B*scattering_blue),   # added thermal population after red readout pulse or device in arm V
                                                        		# from cross correlation in Fig. 2c, see SI section 3 for details

                wcs_tms_ratio = 6.6,                    		# alpha_wcs = np.sqrt(wcs_tms_ratio*scattering_blue*opt_eff)
                
                lifetime_H = 1.3e-6,                   			# mechanical lifetime of device in arm H (seconds)
                lifetime_V = 1.9e-6,                    		# mechanical lifetime of device in arm V (seconds)
                blue_red_delay = 100e-9,                		# time delay between write and read pulses (seconds)
                        
                opt_eff= 0.62*0.2*0.5,                  		# nanobeam optical detection efficiency includeing
                                                       			# - extraction efficency k_e/k,
                                                       			# - fiber coupling efficiency,
                                                       			# - detection filter efficiency,
                                                       			# - optical path efficiency,
     			
                detector_eff = 0.9,                    			# detector efficiency (SSPDs)
                                            
                p_darkcount = 6e-6,                    			# probability to get a click from SSPD darkcount or laser leakage
                interferometer_visibility = 0.99,      			# visibility of the interferometer as from the first order interference of our lasers 
                repetition_time = 20e-6,               			# measurement repetition time (seconds)
                duty_cycle = 0.7,                      			# protocol duty cycle, mostly affected by the periodic 
                                                       			# relocking of the detection filters. Only affects measurement time calculation
                nsigma = 2,                            			# targeted number of standard deviations to show fidelity is above the bound (~2/3)
                                                       			# used to estimate the required measurement time to achieve this

                wcs_in = True,                         			# use weak coherent state as input, if false: use n=1 fock state
                number_resolving_detector = False,     			# whether to assume number resolving detectors
                N = 3,                                 			# Simulated Hilbert space dimension for each mode

            )


logging.getLogger().setLevel('INFO')

used_input_bases = ['H','V','D','L'] 

def get_average_fidelity(measured_fidelities):
    return (measured_fidelities[0] + measured_fidelities[1] + 2* measured_fidelities[2]+2*measured_fidelities[3])/6



# %% ##################################### FIG 3 ################################


parameters = get_simulation_parameters()
measured_fidelities, required_number_of_events, required_measurement_times = teleportation_demonstration(used_input_bases, **parameters)
       
print(f'fidelity = H {measured_fidelities[0]:.2f} V {measured_fidelities[1]:.2f} D {measured_fidelities[2]:.2f} L {measured_fidelities[3]:.2f}')
print("average fidelity", get_average_fidelity(measured_fidelities))



# %% ################################ FIG S3a #################################

bases = ['H','V','D','L'] 

mechanical_lifetimes = np.append(np.linspace(300e-9,2500e-9,15), np.linspace(2600e-9,12e-6,12) )
average_fidelities = np.zeros(len(mechanical_lifetimes))
total_measurement_times = np.zeros(len(mechanical_lifetimes))

for i, lifetime in enumerate(mechanical_lifetimes):
    parameters = get_simulation_parameters()
    parameters['lifetime_H'] = lifetime
    parameters['lifetime_V'] = lifetime
    parameters['repetition_time'] = np.max([7*lifetime,20e-6])  # wait at least 7 times the lifetime for rethermalisation, 
                                                                # with a minimum of 20 us
    measured_fidelities, required_number_of_events, required_measurement_times = teleportation_demonstration(used_input_bases, **parameters)
    average_fidelities[i] = get_average_fidelity(measured_fidelities)
    total_measurement_times[i] = np.sum(required_measurement_times)

plt.figure()    
plt.plot(mechanical_lifetimes,average_fidelities)
plt.xlabel('Mechanical lifetime for both devices')
plt.ylabel(r'Average demonstrated teleportation fidelity')
plt.figure()
plt.plot(mechanical_lifetimes,total_measurement_times)
plt.xlabel('Mechanical lifetime for both devices')
plt.ylabel(r'Required measurment time to achieve an average fidelity 2 sigma above the classical bound')


# %% ##################################### FIG S3b ##############################
pts = 10
wcs_tms_ratios = np.linspace(0.1,20,pts)
average_fidelities = np.zeros(len(wcs_tms_ratios))
total_measurement_times = np.zeros(len(wcs_tms_ratios))

for i, wcs_tms_ratio in enumerate(wcs_tms_ratios):
    parameters = get_simulation_parameters()
    parameters['wcs_tms_ratio'] = wcs_tms_ratio
    measured_fidelities, required_number_of_events, required_measurement_times = teleportation_demonstration(used_input_bases, **parameters)
    average_fidelities[i] = get_average_fidelity(measured_fidelities)
    total_measurement_times[i] = np.sum(required_measurement_times)

plt.figure()    
plt.plot(wcs_tms_ratios,average_fidelities)
plt.xlabel('Ratio WCS probability amplitude and blue scattering probability')
plt.ylabel(r'Average demonstrated teleportation fidelity')
plt.figure()
plt.plot(wcs_tms_ratios,total_measurement_times)
plt.xlabel('Ratio WCS probability amplitude and blue scattering probability')
plt.ylabel(r'Required measurment time to achieve an average fidelity 2 sigma above the classical bound')



# %% ################ Fig. S3c and Fig. S3d thermal population model  ################

   
n_th_blue_H = lambda scattering_blue: n_th_0_A + zeta_inst_A*scattering_blue + 0.03 #to account for increased thermal population due to interfermoeter locking light
n_th_blue_V = lambda scattering_blue: n_th_0_B + zeta_inst_B*scattering_blue + 0.03 #to account for the lock pulse heating
n_th_red_H  = lambda scattering_blue, scattering_red: zeta_delayed_A*scattering_blue + zeta_inst_A * (scattering_red-scattering_blue) 
n_th_red_V  = lambda scattering_blue, scattering_red: zeta_delayed_B*scattering_blue + zeta_inst_B * (scattering_red-scattering_blue) 
    

# %% ##################################### Fig. S3c #################################

pts = 11
ratios = np.linspace(0.3,12, pts)
n_th_sum_H = 0.24 #pm #0.04
n_th_sum_V = 0.10 #pm #0.04

scat_blues =  (n_th_sum_H-n_th_0_A )/(zeta_delayed_A+zeta_inst_A*ratios)
scat_reds = np.array([ scat_blues[i] * ratios[i] for i in range(len(ratios)) ]) 

average_fidelities = np.zeros(len(ratios))
total_measurement_times = np.zeros(len(ratios))

for k,(scat_blue, scat_red) in enumerate(zip(scat_blues, scat_reds)):
    parameters = get_simulation_parameters()
    parameters['scattering_blue'] = scat_blue
    parameters['scattering_red'] = scat_red
    parameters['n_th_blue_H'] = n_th_blue_H(scat_blue)
    parameters['n_th_blue_V'] = n_th_blue_V(scat_blue)
    parameters['n_th_red_H'] = n_th_red_H(scat_blue,scat_red)
    parameters['n_th_red_V'] = n_th_red_V(scat_blue,scat_red)
    measured_fidelities, required_number_of_events, required_measurement_times = teleportation_demonstration(used_input_bases, **parameters)
    average_fidelities[i] = get_average_fidelity(measured_fidelities)
    total_measurement_times[i] = np.sum(required_measurement_times)


plt.figure()
plt.plot(ratios,average_fidelities)
plt.xlabel('Blue/Red scattering probability ratio')
plt.ylabel(r'Average demonstrated teleportation fidelity')
plt.figure()
plt.plot(ratios,total_measurement_times)
plt.xlabel('Blue/Red scattering probability ratio')
plt.ylabel(r'Required measurment time to achieve an average fidelity 2 sigma above the classical bound')



# %% ##################################### Fig. S3d #################################

pts = 21
scat_blues =  np.concatenate(([0.003,0.0032],np.linspace(0.0035,0.06,pts),[0.026,0.027,0.028,0.029]))
scat_reds = 3* scat_blues

average_fidelities = np.zeros(len(scat_blues))
total_measurement_times = np.zeros(len(scat_blues))

for k,(scat_blue, scat_red) in enumerate(zip(scat_blues, scat_reds)):
    parameters = get_simulation_parameters()
    parameters['scattering_blue'] = scat_blue
    parameters['scattering_red'] = scat_red
    parameters['n_th_blue_H'] = n_th_blue_H(scat_blue)
    parameters['n_th_blue_V'] = n_th_blue_V(scat_blue)
    parameters['n_th_red_H'] = n_th_red_H(scat_blue,scat_red)
    parameters['n_th_red_V'] = n_th_red_V(scat_blue,scat_red)
    measured_fidelities, required_number_of_events, required_measurement_times = teleportation_demonstration(used_input_bases, **parameters)
    average_fidelities[i] = get_average_fidelity(measured_fidelities)
    total_measurement_times[i] = np.sum(required_measurement_times)


plt.figure()
plt.plot(scat_blues,average_fidelities)
plt.xlabel('Blue scattering probability')
plt.ylabel(r'Average demonstrated teleportation fidelity')
plt.figure()
plt.plot(scat_blues,total_measurement_times)
plt.xlabel('Blue scattering probability')
plt.ylabel(r'Required measurment time to achieve an average fidelity 2 sigma above the classical bound')




# %% ################################### Fig. 2d #######################################   

pts=21
visibilities=np.zeros(pts)
g2_sames=np.zeros(pts)
g2_diffs=np.zeros(pts)
phis= np.linspace(-np.pi*(1/2+0.2),np.pi*(1/2+0.2),pts)

for i,phi in enumerate(phis):
    parameters = get_simulation_parameters()
    parameters['phi'] = phi
    visibilities[i], g2_sames[i],g2_diffs[i],_ = entanglement_simulation(**parameters)

plt.figure()  
plt.plot(phis/np.pi,g2_sames)
plt.plot(phis/np.pi,g2_diffs)
plt.xlabel('phase difference')
plt.ylabel(r'$g^{(2)}$')

# %% ##################################### Fig. S6 #######################################

pts=120
visibilities = np.zeros(pts)
g2_sames=np.zeros(pts)
g2_diffs=np.zeros(pts)
delays= np.linspace(10e-9,1.3e-6,pts)

delta_mech = 8.22086e6 *2 * np.pi /2 #because adding to both blue and red

gamma_rise_H = 410 #delayed heating time constant, ref. Fig. S2a
gamma_rise_V = 200 #delayed heating time constant, ref. Fig. S2a

for i,delay in enumerate(delays):
    parameters = get_simulation_parameters()
    parameters['phi'] = delay*delta_mech + np.pi/4 - np.pi/32
    parameters['blue_red_delay'] = delay
    parameters['n_th_red_H'] *= (3 - 2*np.exp(-(delay-100e-9)/gamma_rise_H)) # accounting for additional delayed heating
    parameters['n_th_red_V'] *= (3 - 2*np.exp(-(delay-100e-9)/gamma_rise_V)) # accounting for additional delayed heating
      
    visibilities[i], g2_sames[i],g2_diffs[i],_ = entanglement_simulation(**parameters)

plt.figure()  
plt.plot(delays,g2_sames)
plt.plot(delays,g2_diffs)
plt.xlabel('Delay red blue (s)')
plt.ylabel(r'$g^{(2)}$')
