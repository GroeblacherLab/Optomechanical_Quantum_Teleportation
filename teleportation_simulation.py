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

Main simulation function to simulate the teleportation demonstration
experiement from:

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
import numpy as np
import qutip
from quantum_optics_lib import QuantumOpticsSim

def teleportation_demonstration(bases = ['H','V','D','A','L','R'], **kw):
        measured_fidelities = np.zeros(len(bases))
        required_number_of_events = np.zeros(len(bases))
        required_measurement_times = np.zeros(len(bases))
        for i,basis in enumerate(bases):
            logging.info(f'Basis {basis}')
               
            if basis == 'H':
                theta_in      = 0 
                phi_in        = 0
                theta_readout = 0
                phi_readout   = 0
            elif basis=='V':
                theta_in      = np.pi/4 
                phi_in        = 0
                theta_readout = np.pi/4
                phi_readout   = 0
            elif basis=='D':
                theta_in      = np.pi/8
                phi_in        = np.pi/4
                theta_readout = np.pi/8
                phi_readout   = 0
            elif basis=='A':
                theta_in      = -np.pi/8
                phi_in        = np.pi/4
                theta_readout = -np.pi/8
                phi_readout   = 0
            elif basis=='L':
                theta_in      = 0
                phi_in        = np.pi/4
                theta_readout = np.pi/8
                phi_readout   = np.pi/4
            elif basis=='R':
                theta_in      = 0
                phi_in        = -np.pi/4
                theta_readout = -np.pi/8
                phi_readout   = np.pi/4
            else:
                raise Exception('Unknown input basis setting')
            kw['theta_in'] = theta_in
            kw['phi_in'] = phi_in
            kw['theta_readout'] = theta_readout
            kw['phi_readout'] = phi_readout

            measured_fidelities[i], required_number_of_events[i], required_measurement_times[i] = teleportation_simulation(**kw)
        
        return measured_fidelities, required_number_of_events, required_measurement_times


def teleportation_simulation(
         # these are dummy default values, see simulation_main for actural values used
       
        scattering_blue = 0.001,                # blue pulse scattering probability for each device 
        scattering_red = 0.001,                 # red readout pulse scattering probability for each device
        
        n_th_blue_H = 0,                        # initial thermal population of mechanical mode
        n_th_blue_V = 0,                        # initial thermal population of mechanical mode
        n_th_red_H = 0,                         # added thermal population after red readout pulse
        n_th_red_V = 0,                         # added thermal population after red readout pulse

        wcs_tms_ratio = 10,                     # alpha_wcs = np.sqrt(wcs_tms_ratio*scattering_blue*opt_eff)

        lifetime_H = 1e-6,                      # mechanical lifetime of device in arm H (seconds)
        lifetime_V = 1e-6,                      # mechanical lifetime of device in arm V (seconds)

        blue_red_delay = 100e-9,                # time delay between write and read pulses (seconds)
                
        opt_eff= 1,                             # nanobeam optical detection efficiency includeing
                                                # - extraction efficency k_e/k,
                                                # - fiber coupling efficiency,
                                                # - detection filter efficiency,
                                                # - optical path efficiency,

        detector_eff = 1,                       # SNSPD detector efficiency 
                                    
        p_darkcount = 0,                        # probability to get a click from SNSPD darkcount or laser leakage
        interferometer_visibility = 1,          # visibility of the interferometer as from the first order interference of our lasers 
        
        repetition_time = 100e-6,               # measurement repetition time (seconds)
        duty_cycle = 1,                         # measurment duty cycle, mostly affected by the periodic 
                                                # relocking of the detection filters. Only affects measurement time calculation
        nsigma = 2,                             # targeted number of standard deviations to show fidelity is above the bound (~2/3)

        coincidence_type = 0,                   # [[H_1 V_1],[H_1 V_2],[V_1 H_2],[H_2 V_2]] for 50/50 BS outputs 
        wcs_in = True,                          # use weak coherent state as input, if false: use n=1 fock state
        number_resolving_detector = False,      # whether to assume number resolving detectors
        N = 3,                                  # Simulated Hilbert space dimension for each mode

        theta_in      = 0,                      # teleportation input half-waveplate angle
        phi_in        = 0,                      # teleportation input quarter-waveplate angle
        theta_readout = 0,                      # teleportation readout half-waveplate angle
        phi_readout   = 0,                      # teleportation readout quarter-waveplate angle
        ):                      
   
    alpha_wcs = np.sqrt(wcs_tms_ratio*scattering_blue*opt_eff)
    logging.debug(f'alpha_wcs = {alpha_wcs}')

    modes          = ['mec_H',
                      'mec_V',
                      'opt_H',
                      'opt_V',
                      'wcs_H',
                      'wcs_V']

    qsim = QuantumOpticsSim(N,modes)
    
    initial_in_state = qutip.coherent_dm(N, alpha_wcs) if wcs_in else qutip.fock_dm(N,1)
    
    initial_states = [qutip.thermal_dm(N, n_th_blue_H),
                      qutip.thermal_dm(N, n_th_blue_V),
                      qutip.coherent_dm(N, 0.),
                      qutip.coherent_dm(N, 0.),
                      initial_in_state,
                      qutip.coherent_dm(N, 0.)]

    rho_0 = qutip.tensor(*initial_states)

    #1: Prepre WCS input state
    rho_1hw  =   qsim.half_wave_plate(rho_0,  'wcs_H', 'wcs_V', theta_in)
    rho_1qw = qsim.quarter_wave_plate(rho_1hw, 'wcs_H', 'wcs_V', phi_in)
    rho_1=rho_1qw

    qsim.print_expectation(rho_1,'rho_1')

    #2: Prepare 2x two-mode-squeezed state for H and V optomechanical device
    rho_2H = qsim.two_mode_squeezing(rho_1, 'mec_H','opt_H',2*np.sqrt(scattering_blue))
    rho_2V = qsim.two_mode_squeezing(rho_2H,'mec_V','opt_V',2*np.sqrt(scattering_blue))#check
    rho_2  = rho_2V

    qsim.print_expectation(rho_2,'rho_2')
    
    #2b interferometer dephasing
    rho_2b = qsim.dephase(rho_2,'opt_V',interferometer_visibility)
    
    qsim.print_expectation(rho_2b,'rho_2b')
    
    #2c photon loss
    rho_2cH = qsim.loss(rho_2b, 'opt_H',opt_eff)
    rho_2cV = qsim.loss(rho_2cH,'opt_V',opt_eff)   
    rho_2c  = rho_2cV
    

    rho_2d = qsim.phase(rho_2c,   'opt_H', phi_readout)#XXX
    rho_2e  =   qsim.half_wave_plate(rho_2d, 'opt_H', 'opt_V', 0)
    qsim.print_expectation(rho_2e,'rho_2e')
    
    #3: combine wcs and opt on 50/50 beam splitter
    rho_3H = qsim.beam_splitter(rho_2e,'wcs_H','opt_H')
    rho_3V = qsim.beam_splitter(rho_3H,'wcs_V','opt_V')
    rho_3=rho_3V

    qsim.print_expectation(rho_3,'rho_3')

    #4: Measure coincidences
    coincidences = [['opt_H','opt_V'],['opt_H','wcs_V'],['wcs_H','opt_V'],['wcs_H','wcs_V']]
    coincidence_modes = coincidences[coincidence_type]

    rho_darkcount_1,p_click_1 = qsim.single_click(rho_3,coincidence_modes[0],number_resolving_detector=number_resolving_detector)
    rho_darkcount_2,p_click_2 = qsim.single_click(rho_3,coincidence_modes[1],number_resolving_detector=number_resolving_detector)
    rho_succes,p_coincidence = qsim.coincidence(rho_3,coincidence_modes[0],coincidence_modes[1],number_resolving_detector=number_resolving_detector)
   
    if wcs_in:
        logging.debug(f'p_coincidence_click {p_coincidence*detector_eff**2:.2e}, expected \
                     {wcs_tms_ratio*scattering_blue**2*opt_eff**2*detector_eff**2:.2e}')

    logging.debug(f'rho_succes norm {rho_succes.norm()}')
    

    # Include effect of dark counts #approximate, for p_click <<1, (i.e. a non-click gives 0 information)
    rho_4 = (p_coincidence*detector_eff**2*rho_succes + \
                p_click_1*detector_eff*p_darkcount*rho_darkcount_1 + \
                p_click_2*detector_eff*p_darkcount*rho_darkcount_2)/ \
             (p_coincidence*detector_eff**2 + p_click_1*detector_eff*p_darkcount +  p_click_2*detector_eff*p_darkcount)
    #approximate, for p_coincidence, p_dc,p_click <<1
    qsim.print_expectation(rho_4,'rho_4')
    logging.debug(f'rho4 norm {rho_4.norm()}')

    #5: trace out optical modes, keep only mechanics
    rho_mec = rho_4.ptrace([0,1])
    logging.debug(f'rho_mec norm {rho_mec.norm()}')

    rho_5_list = [rho_mec]
    for m in modes[2:]:
        rho_5_list.append(qutip.fock_dm(N, 0)) #expliciteloy reset to vacuum
    rho_5 = qutip.tensor(*rho_5_list)

    #5d phonon loss
    rho_5dH = qsim.loss(rho_5, 'mec_H',np.exp(-blue_red_delay/lifetime_H))
    rho_5dV = qsim.loss(rho_5dH,'mec_V',np.exp(-blue_red_delay/lifetime_V))  
    rho_5 = rho_5dV
    
    #5b couple to thermal bath here to introduce the n_th using a faster alternative, by reusing existing hilbert space
    H_tms_H = qutip.Qobj.expm( 1j*np.sqrt(n_th_red_H) * (qsim.a['mec_H'].dag()*qsim.a['wcs_H'].dag() + qsim.a['mec_H']*qsim.a['wcs_H']))
    H_tms_V = qutip.Qobj.expm( 1j*np.sqrt(n_th_red_V) * (qsim.a['mec_V'].dag()*qsim.a['wcs_V'].dag() + qsim.a['mec_V']*qsim.a['wcs_V']))
    rho_5b = H_tms_V.dag() * H_tms_H.dag() * rho_5 * H_tms_H * H_tms_V
    logging.debug(f'rho_5b norm {rho_5b.norm()}')
    rho_5b_mec = rho_5b.ptrace([0,1])
    logging.debug(f'rho_5b_mec norm {rho_5b_mec.norm()}')
    rho_5b_list = [rho_5b_mec]
    for m in modes[2:]:
        rho_5b_list.append(qutip.fock_dm(N, 0)) #expliciteloy reset to vacuum
    rho_5c = qutip.tensor(*rho_5b_list)
    
    qsim.print_expectation(rho_5,'rho_5')
    
    #6: read out mechanics
    rho_6H = qsim.beam_splitter(rho_5c,'mec_H','opt_H',theta = 2*np.sqrt(scattering_red))
    rho_6V = qsim.beam_splitter(rho_6H,'mec_V','opt_V',theta = 2*np.sqrt(scattering_red))
    rho_6 = rho_6V

    qsim.print_expectation(rho_6,'rho_6')

    #7: rotate onto correct basis to determine overlap with input state
    rho_7qw = qsim.phase(rho_6,   'opt_H', phi_readout)#phi_readout)XXX
    rho_7hw  =   qsim.half_wave_plate(rho_7qw, 'opt_H', 'opt_V', theta_readout)
    rho_7 = rho_7hw

    qsim.print_expectation(rho_7,'rho_7')
    
    #7b photon loss
    rho_7bH = qsim.loss(rho_7,  'opt_H',opt_eff)
    rho_7bV = qsim.loss(rho_7bH,'opt_V',opt_eff)   
    rho_7b  = rho_7bV
    

    #8: measure H/V part
    _,p_click_H = qsim.single_click(rho_7b,'opt_H',number_resolving_detector=number_resolving_detector)
    _,p_click_V = qsim.single_click(rho_7b,'opt_V',number_resolving_detector=number_resolving_detector)
    
    
    logging.debug(f'p_red_click = {(p_click_H*detector_eff+p_click_V*detector_eff):.2e}, expected {scattering_red*opt_eff*detector_eff:.2e}')
    
    p_click_total = (p_click_H*detector_eff+p_click_V*detector_eff)
    p_success = p_coincidence*detector_eff**2*p_click_total*4
    # total p_succes is *4, because we simulate only measure one of succesfull 4 coincidence_types
    
    F_measured_raw = 1-p_click_H/(p_click_H+p_click_V) # 1- because we always measure the antisymetric Bell state
    F_measured = (F_measured_raw*p_click_total + 0.5*p_darkcount)/(p_click_total+p_darkcount) # effect of readout dark counts
    
    F_bound = 2/3 # bound (for alpha_wcs<<1)

    # number of events needed to get > nsigma standard deviations
    n_required=int(np.ceil((F_measured*nsigma**2 - F_measured**2*nsigma**2)/(F_measured-F_bound)**2)) if F_measured>F_bound else np.inf

     #some measurement time effects.
    succes_time=1/(p_success*duty_cycle)*repetition_time

    measurement_time_required = succes_time*n_required

    logging.info(f'{repetition_time}, {succes_time}')
    logging.info(f'F_measured = {F_measured:.3f},  n_required = {n_required},p_success = {p_success:.3e}, meas_time = {measurement_time_required/60/60/24:.1f} days')

    return F_measured, n_required, measurement_time_required
