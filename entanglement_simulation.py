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

Main simulation function to simulate the etanglement experiement in the 
teleportation setup from:

Optomechanical quantum teleportation
Niccol`o Fiaschi,1,∗ Bas Hensen,1,∗ Andreas Wallucks,1 Rodrigo
Benevides,1,2 Jie Li,1,3 Thiago P. Mayer Alegre,2 and Simon Gröblacher1,

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


def entanglement_simulation(
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
              
        theta = np.pi/2,                        # half wave plate angle
        phi = 0,                                # quarter wave plate angle
        click_mode      = 'opt_H',              # first (heralding) click mode
        diff_click_mode = 'opt_V',              # other detector click mode
        number_resolving_detector = False,      # whether to assume number resolving detectors
        N = 3,                                  # Simulated Hilbert space dimension for each mode
        **kw):
              
    modes          = ['mec_H',
                      'mec_V',
                      'opt_H',
                      'opt_V']

    qsim = QuantumOpticsSim(N,modes)

    initial_states = [qutip.thermal_dm(N, n_th_blue_H),
                      qutip.thermal_dm(N, n_th_blue_V),
                      qutip.coherent_dm(N, 0.),
                      qutip.coherent_dm(N, 0.)]
    rho_0 = qutip.tensor(*initial_states)

    #1: Prepare 2x two-mode-squeezed state for H and V optomechanical device
    rho_1H = qsim.two_mode_squeezing(rho_0, 'mec_H','opt_H',2*np.sqrt(scattering_blue))
    rho_1V = qsim.two_mode_squeezing(rho_1H,'mec_V','opt_V',2*np.sqrt(scattering_blue))
    rho_1  = rho_1V

    qsim.print_expectation(rho_1,'rho_1')
    
    #1b interferometer dephasing
    rho_1b = qsim.dephase(rho_1,'opt_V',interferometer_visibility)

    qsim.print_expectation(rho_1b,'rho_1b')

    rho_out,p_blue_click = detect(rho_1b,qsim,click_mode,theta,phi,opt_eff,detector_eff,p_darkcount,number_resolving_detector)
    logging.debug(f'p_blue_click = {(p_blue_click*detector_eff):.2e}, expected {scattering_blue*opt_eff*detector_eff:.2e}')

    #4: trace out optical modes, keep only mechanics
    rho_mec = rho_out.ptrace([0,1])
    logging.debug(f'rho_mec norm {rho_mec.norm()}')
    rho_mec_norm = rho_1b.ptrace([0,1])
    
    rho_4 = rho_from_mec(rho_mec, modes, N)
    rho_4_norm = rho_from_mec(rho_mec_norm, modes, N)
    
    #5d phonon loss
    rho_4dH = qsim.loss(rho_4, 'mec_H',np.exp(-blue_red_delay/lifetime_H))
    rho_4dV = qsim.loss(rho_4dH,'mec_V',np.exp(-blue_red_delay/lifetime_V))  
    rho_4 = rho_4dV

    rho_4dH_norm = qsim.loss(rho_4_norm, 'mec_H',np.exp(-blue_red_delay/lifetime_H))
    rho_4dV_norm = qsim.loss(rho_4dH_norm,'mec_V',np.exp(-blue_red_delay/lifetime_V))  
    rho_4_norm = rho_4dV_norm
    
    #4b adding the thermal population to the mechanical mode using a faster alternative, by reusing existing hilbert space
    H_tms_H = qutip.Qobj.expm( 1j*np.sqrt(n_th_red_H) * (qsim.a['mec_H'].dag()*qsim.a['opt_H'].dag() + qsim.a['mec_H']*qsim.a['opt_H']))
    H_tms_V = qutip.Qobj.expm( 1j*np.sqrt(n_th_red_V) * (qsim.a['mec_V'].dag()*qsim.a['opt_V'].dag() + qsim.a['mec_V']*qsim.a['opt_V']))
    rho_4b = H_tms_V.dag() * H_tms_H.dag() * rho_4 * H_tms_H * H_tms_V
    logging.debug(f'rho_4b norm {rho_4b.norm()}')
    
    rho_4b_mec = rho_4b.ptrace([0,1])
    rho_4c = rho_from_mec(rho_4b_mec, modes, N)
    
    rho_4b_norm = H_tms_V.dag() * H_tms_H.dag() * rho_4_norm * H_tms_H * H_tms_V
    rho_4b_mec_norm = rho_4b_norm.ptrace([0,1])
    rho_4c_norm = rho_from_mec(rho_4b_mec_norm, modes, N)

    #5: read out mechanics
    rho_5H = qsim.beam_splitter(rho_4c,'mec_H','opt_H',theta = 2*np.sqrt(scattering_red))
    rho_5V = qsim.beam_splitter(rho_5H,'mec_V','opt_V',theta = 2*np.sqrt(scattering_red))
    rho_5 = rho_5V

    rho_5H_norm = qsim.beam_splitter(rho_4c_norm,'mec_H','opt_H',theta = 2*np.sqrt(scattering_red))
    rho_5V_norm = qsim.beam_splitter(rho_5H_norm,'mec_V','opt_V',theta = 2*np.sqrt(scattering_red))
    rho_5_norm = rho_5V_norm

    qsim.print_expectation(rho_5,'rho_5')

    #6 detection part
    _,p_click_same = detect(rho_5,qsim, click_mode,     theta,phi,opt_eff,detector_eff,p_darkcount,number_resolving_detector)
    _,p_click_diff = detect(rho_5,qsim, diff_click_mode,theta,phi,opt_eff,detector_eff,p_darkcount,number_resolving_detector)
    
    _,p_click_same_norm = detect(rho_5_norm,qsim, click_mode,      theta,phi,opt_eff,detector_eff,p_darkcount,number_resolving_detector)
    _,p_click_diff_norm = detect(rho_5_norm,qsim, diff_click_mode, theta,phi,opt_eff,detector_eff,p_darkcount,number_resolving_detector)
    
    g2_same = p_blue_click*detector_eff*p_click_same*detector_eff/(p_blue_click*detector_eff*p_click_same_norm*detector_eff)
    g2_diff = p_blue_click*detector_eff*p_click_diff*detector_eff/(p_blue_click*detector_eff*p_click_diff_norm*detector_eff)
  
    logging.debug(f'p_red_click = {(p_click_same*detector_eff+p_click_diff*detector_eff):.2e}, expected {scattering_red*opt_eff*detector_eff:.2e}')
    logging.debug(f'p_red_click_norm = {(p_click_same_norm*detector_eff+p_click_diff_norm*detector_eff):.2e}, expected {scattering_red*opt_eff*detector_eff:.2e}')
    
    p_success = p_blue_click*detector_eff*(p_click_same*detector_eff+p_click_diff*detector_eff) *2 
    # total p_succes is *2, because we only measure one of 2 click_mode

    #calculate the visibility expected
    V = np.abs((g2_same - g2_diff)/(g2_same + g2_diff))
    
    logging.info(f'g2_same = {g2_same:.2f}, g2_diff = {g2_diff:.2f}, p_success = {p_success:.3e}')
    logging.info(f'V = {V:.2f}')
    return V, g2_same, g2_diff, p_success

    
def detect(rho_in,qsim,click_mode,theta,phi,opt_eff,detector_eff,p_darkcount,number_resolving_detector):
    
     #1c photon loss
    rho_1cH = qsim.loss(rho_in, 'opt_H',opt_eff)
    rho_1cV = qsim.loss(rho_1cH,'opt_V',opt_eff)   
    rho_1c  = rho_1cV

    qsim.print_expectation(rho_1c,'rho_1c')

    # 2 eom for the phase sweep and waveplate to let the photons from the devices interfere
    rho_2qw = qsim.phase(rho_1c, 'opt_H', phi)
    rho_2hw = qsim.half_wave_plate(rho_2qw, 'opt_H', 'opt_V', theta/4)
    rho_2 = rho_2hw

    qsim.print_expectation(rho_2,'rho_2')
    
    #3: Measure coincidences

    rho_succes,p_click = qsim.single_click(rho_2,click_mode,number_resolving_detector=number_resolving_detector)
    
    # Include effect of dark coutns
    rho_no_click = rho_2
    rho_out = (p_click*detector_eff*rho_succes + \
             p_darkcount*rho_no_click)/ \
             (p_click*detector_eff + p_darkcount) 
    #approximate, for, p_dc,p_click << 1
    qsim.print_expectation(rho_out,'rho_out')
    logging.debug(f'rho_out norm {rho_out.norm()}')

    return rho_out,p_click

def rho_from_mec(rho_mec, modes, N):
    #explicitely resets photonic modes to vacuum
    rho_list = [rho_mec]
    for m in modes[2:]:
        rho_list.append(qutip.fock_dm(N, 0)) 
    return qutip.tensor(*rho_list)
