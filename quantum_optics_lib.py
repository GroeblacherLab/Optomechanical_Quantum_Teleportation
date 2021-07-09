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

Common library used for numerical simulations in:

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
import numpy as np
from scipy.special import comb
from scipy.sparse import dia_matrix
import qutip
import logging

class MultimodeSim():
    def __init__(self,N,modes):
        #operators:
        self.N = N
        self.modes = modes # list of mode names
        
        self.a = {} #put anihilation operator for each mode in a dictionary ie a_mode1 = a['mode1'] = qutip.tensor([detroy(N),qeye(N),qeye(N),qeye(N),qeye(N),qeye(N)])
        for m in modes:
             self.a[m] = self.make_single_mode_operator(m,qutip.destroy(N))

    def make_single_mode_operator(self,mode,op):
        return self.make_tensor_operator([mode],[op])

    def make_tensor_operator(self,target_modes,operators):
        """
         returns qutip.tensor([qeye(N),qeye(N), ... ,qeye(N),qeye(N)]),
        (i.e. the identity operator in the Hilbert space defined by the tensor
        product of all hilbert spaces of each mode in "all_modes"), but where
        for each of the "target_modes" the qeye is replaced by the corresponding operator
        in "operators" ("target_modes" and "operators" are lists of equal length).
        """

        op_list = [qutip.qeye(self.N)]*len(self.modes)
        for op,m in zip(operators,target_modes):
            op_list[self.modes.index(m)] = op
        return qutip.tensor(*op_list)

    def expect_g2(self,rho,mode):
        n = self.expect_n(rho,mode)
        if abs(n)>0:
            return qutip.expect(self.a[mode].dag()*self.a[mode].dag()*self.a[mode]*self.a[mode], rho) / n**2
        else:
            return np.NaN

    def expect_n(self,rho,mode):
        return qutip.expect(self.a[mode].dag()*self.a[mode], rho)

    def print_expectation(self,rho,rho_name):
        logging.debug('-'*10 + rho_name + '-'*10)
        for m in self.modes:
            n = self.expect_n(rho,m)
            g2 = self.expect_g2(rho,m)

            logging.debug(f"n_{m} = {n:.4f}, g^(2)_{m} = {g2:.2f}")
        logging.debug('-'*(20+len(rho_name)))

    def measure_single_mode(self,rho,mode,projection_operator):
        return self.measure(rho,[mode],[projection_operator])

    def measure(self,rho,target_modes,projection_operators):
        P = self.make_tensor_operator(target_modes,projection_operators)
        p = np.real((P*rho).tr()) #remove small nonzero imaginary part
        if p>0:
            return P*rho*P.dag()/p,p
        else:
            return rho,p

class QuantumOpticsSim(MultimodeSim):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.P0 = qutip.basis(self.N,0)*qutip.basis(self.N,0).dag()
        self.P1 = qutip.basis(self.N,1)*qutip.basis(self.N,1).dag()
        self.Pnot0 = qutip.qeye(self.N)-self.P0

    def single_click(self,rho,mode,number_resolving_detector=False):
        Pclick = self.P1 if number_resolving_detector else self.Pnot0
        return self.measure_single_mode(rho,mode,Pclick)

    def coincidence(self,rho,mode1,mode2,number_resolving_detector=False):
        Pclick = self.P1 if number_resolving_detector else self.Pnot0
        return self.measure(rho,[mode1,mode2],[Pclick,Pclick])

    def half_wave_plate(self,rho,mode_H,mode_V,theta):
        Sy = -1j * (self.a[mode_H].dag()*self.a[mode_V] - self.a[mode_H]*self.a[mode_V].dag())
        U = qutip.Qobj.expm( -1j/2*theta*4 * Sy)
        return U.dag()* rho * U

    def quarter_wave_plate(self,rho,mode_H,mode_V,phi):
        Sx = self.a[mode_H].dag()*self.a[mode_V] + self.a[mode_H]*self.a[mode_V].dag()
        U = qutip.Qobj.expm( 1j/2*phi*2 * Sx)
        return U.dag()* rho * U

    def beam_splitter(self,rho,mode1,mode2, theta = np.pi / 2):
        U = qutip.Qobj.expm( 1j/2*theta * (self.a[mode1].dag()*self.a[mode2] + self.a[mode1]*self.a[mode2].dag()))
        return U.dag()* rho * U

    def two_mode_squeezing(self,rho,mode1,mode2,theta):#theta = 2*np.sqrt(scat_prob)
        U = qutip.Qobj.expm( 1j/2*theta * (self.a[mode1].dag()*self.a[mode2].dag() + self.a[mode1]*self.a[mode2]))
        return U.dag()* rho * U

    def phase(self,rho,mode,phi):
        U = qutip.Qobj.expm( 1j * phi * self.a[mode].dag()*self.a[mode])
        return U.dag()* rho * U

    def dephase(self,rho,mode,visibility):
        return (1/2+1/2*visibility) * rho + (1/2-1/2*visibility) * self.phase(rho,mode,np.pi)

    def loss(self,rho,mode,eta):
        Aks = get_loss_krauss_operators(eta,self.N)
        krauss_ops = [self.make_single_mode_operator(mode,qutip.Qobj(Ak)) for Ak in Aks] 
        return sum([b*rho*b.dag() for b in krauss_ops])

    def couple_thermal_bath(self, rho, mode, n_th): 
        """
		couple the mode to a thermal bath:
		first use the two mode squeeze interaction (with amplitude as the thermal occupation n_th) with a enviroment state
		then trace out the enviroment
        """ 
        rho_env = qutip.fock_dm(self.N,0)
        rho_tot = qutip.tensor(rho,rho_env)
        a_env = qutip.tensor(self.a[mode]**0,qutip.destroy(self.N))
        a_sys = qutip.tensor(self.a[mode],qutip.qeye(self.N))
        H_tms = qutip.Qobj.expm( 1j*np.sqrt(n_th) * (a_sys.dag()*a_env.dag() + a_sys*a_env))
        rho_tot_new = H_tms.dag() * rho_tot * H_tms
        keep_list=np.arange(len(rho.dims[0])).tolist()
        return rho_tot_new.ptrace(keep_list)
    
def get_loss_krauss_operators(eta,N):
    # See for example eq (13) in Liu, Y.; Özdemir, Ş. K.; Miranowicz, A.; Imoto, N. 
    # Kraus Representation of a Damped Harmonic Oscillator and Its Application. 
    #Phys. Rev. A 2004, 70 (4), 042308. https://doi.org/10.1103/PhysRevA.70.042308.
    
    Nr=np.arange(N, dtype=np.int)
    Aks=[]
    for k in range(N):
        factors = np.sqrt(comb(Nr,k)) * (eta)**((Nr-k)/2) * (1-eta)**(k/2)
        offset = k
        Ak=dia_matrix((factors, [offset]), shape=(N, N))
        Aks.append(Ak)
    
    # Check sum of Kraus operators is identity:    
    logging.debug(sum([Ak.transpose().conjugate()*Ak for Ak in Aks]).toarray())
    
    return Aks
  
