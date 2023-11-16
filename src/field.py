import numpy as np
from astropy import units as U
from scipy.signal import convolve2d

class Field:
    def __init__(self, objects, pertubations, telescope_responses, wavelength):
        self.objects = objects if isinstance(objects, list) else [objects]
        self.pertubations = pertubations if isinstance(pertubations, list) else [pertubations]
        self.telescope_responses = telescope_responses if isinstance(telescope_responses, list) else [telescope_responses]
        self.wavelength = wavelength
  
    def __call__(self, u, v):
        assert u.shape == v.shape, "phi and theta must have the same shape"
        sky = self.objects[0](u, v)
        for obj in self.objects[1:]:
            sky += obj(u, v) * U.cd
        for pertubation in self.pertubations:
            sky *= pertubation(u, v)
        for response in self.telescope_responses:
            sky = convolve2d(sky.value, response(u, v), mode='same') * sky.unit
        return sky
    
class Element:
    # Must be expressed in cd

    @staticmethod
    def white(phi,theta):
        white = np.zeros_like(phi, dtype=complex)
        white += (1+0j)
        return white

    @staticmethod
    def unresolved_star(phi,theta):
        star = np.zeros_like(phi, dtype=complex)
        star[len(star)//2,len(star)//2] = 1+0j
        return star
    
    @staticmethod
    def resolved_star(phi,theta):
        R = np.sqrt(phi**2+theta**2)
        star = (R < 0.01).astype(int)+0j
        return star * U.cd

    @staticmethod
    def planet(phi,theta):
        R = np.sqrt(phi**2+theta**2)
        R = np.roll(R, (2,1), axis=(0,1))
        planet_object = (R < 0.001).astype(int)+0j
        return planet_object * 1e-6

class Archetype:

    @staticmethod
    def uniform_disk(phi=0*U.mas, theta=0*U.mas, angular_diameter=1*U.mas, intensity=1*U.cd):
        def uniform_disk(u, v):
            R = np.sqrt((u - phi).to(U.rad)**2+(v-theta).to(U.rad)**2)
            return (R < angular_diameter.to(U.rad)/2).astype(int) * intensity
        return uniform_disk

class Perturbation:

    @staticmethod
    def atmosphere(phi, theta):
        phase_shift = np.exp(1j*np.random.rand(*phi.shape)*0.1*2*np.pi)
        return phase_shift
    
class TelescopeResponse:
    
    @staticmethod
    def PSF(phi, theta):
        PSF = np.zeros_like(phi, dtype=complex)
        PSF = (np.sqrt(phi**2+theta**2) < 0.01).astype(int)+0j
        PSF = np.fft.fftshift(np.fft.fft2(PSF))
        return PSF / np.max(PSF)