import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from .discrete_field import DiscretField

class Telescope:
    all = []

    def __init__(self, x, y, image_size=100, field_width=1*u.arcsec):
        # position relative to the origin of the interferometer
        self.Bx = x 
        self.By = y
        
        # image size and resolution
        self.image_size = image_size
        self.field_width = field_width
        self.resolution = self.field_width / self.image_size
        
        # compute the phase map
        self.u_map, self.v_map = np.meshgrid(np.arange(self.image_size)-self.image_size//2, np.arange(self.image_size)-self.image_size//2)
        self.u_map = self.u_map * self.resolution
        self.v_map = self.v_map * self.resolution

        Telescope.all.append(self)

    def observe(self, field, wavelength):
        self.phase_map = (self.Bx * np.sin(self.u_map.to(u.rad).value) + self.By * np.sin(self.v_map.to(u.rad).value)) / wavelength
        self.field = DiscretField(field(self.u_map, self.v_map) * np.exp(1j * self.phase_map.value))

    #===========================================================================
    # Plotting
    #===========================================================================

    @staticmethod
    def plot_position():
        plt.figure()
        for i, t in enumerate(Telescope.all):
            plt.scatter(t.Bx.value, t.By.value, label=f"Téléscope {i+1}")
        plt.title("Telescopes positions (m)")
        plt.legend()
        plt.show()

    def plot_intensity(self):
        plt.imshow(self.field.intensity.value, cmap="inferno")
        plt.title("Observed sky")
        plt.colorbar()
        plt.show()

    @staticmethod
    def plot_phases():
        plt.gcf().suptitle("Relative phase according to the position on the sky", fontsize=14)
        for i, t in enumerate(Telescope.all):
            plt.subplot(2,2,i+1)
            plt.imshow(np.angle(t.field.complex.value))
            cbar = plt.colorbar(ticks=[-np.pi, 0, np.pi])
            cbar.ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])  # vertically oriented colorbar
            plt.title(f"Telescope {i+1}")
        plt.show()