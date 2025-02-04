import numpy as np
import numba as nb
from astropy import units as u
import matplotlib.pyplot as plt
from io import BytesIO
import ipywidgets as widgets
from IPython.display import clear_output, display

# UT position -----------------------------------------------------------------

def get_VLTI_relative_positions() -> np.ndarray[u.Quantity]:
    """
    Get the relative position of the UTs, in meter.

    Returns
    -------
    - Array of the 4 UT positions in the plane perpendicular to the line of sight (4,2) [m]
    """

    # UT coordinates... obtained on Google map 😅 <- TODO: update with precise positions
    UT_pos = np.array([
        [-70.4048732988764, -24.627602893919807],
        [-70.40465753243652, -24.627118902835786],
        [-70.40439460074228, -24.62681028261176],
        [-70.40384287956437, -24.627033500373024]
    ]) # ⚠️ Expressed in [longitude, latitude] to easily convert to [x, y]
    
    # We are only interested in the relative positions of the UTs
    # The first UT is the reference
    UT_pos -= UT_pos[0]

    # Altitude of the UTs
    earth_radius = 6_378_137 * u.m
    UTs_elevation = 2_635 * u.m

    # Angle to distance conversion
    return np.tan((UT_pos * u.deg).to(u.rad)) * (earth_radius + UTs_elevation)

# Projected position ----------------------------------------------------------

def project_position(
        r: u.Quantity,
        h: u.Quantity,
        l: u.Quantity,
        δ: u.Quantity,
    ) -> u.Quantity:
    """
    Project the telescope position in a plane perpendicular to the line of sight.

    Parameters
    ----------
    - r: Array of telescope positions
    - h: Hour angle
    - l: Latitude
    - δ: Declination

    Returns
    -------
    - Array of projected telescope positions (same shape and unit as p)
    """
    h = h.to(u.rad)
    l = l.to(u.rad)
    δ = δ.to(u.rad)
    
    return project_position_njit(r.value, h, l, δ) * r.unit

@nb.njit()
def project_position_njit(
        r: np.ndarray[float],
        h: float,
        l: float,
        δ: float,
    ) -> np.ndarray[float]:
    """
    Project the telescope position in a plane perpendicular to the line of sight.

    Parameters
    ----------
    - r: Array of telescope positions
    - h: Hour angle (in radian)
    - l: Latitude (in radian)
    - δ: Declination (in radian)

    Returns
    -------
    - Array of projected telescope positions (same shape and unit as p)
    """

    M = np.array([
        [ -np.sin(l)*np.sin(h),                                np.cos(h)          ],
        [ np.sin(l)*np.cos(h)*np.sin(δ) + np.cos(l)*np.cos(δ), np.sin(h)*np.sin(δ)],
    ])

    r_projected = np.empty_like(r)
    for i, pos in enumerate(r):
        r_projected[i] = M @ np.flip(pos)

    return r_projected

# Plotting --------------------------------------------------------------------

def plot_positions(
        r: np.ndarray[u.Quantity],
        h: u.Quantity,
        Δh: u.Quantity,
        l: u.Quantity,
        δ: u.Quantity,
        N:int = 11,
        return_image = False,
    ):
    """
    Plot the telescope positions over the time.

    Parameters
    ----------
    - r: Array of telescope positions
    - h: Hour angle
    - Δh: Hour angle range
    - l: Latitude
    - δ: Declination
    - N: Number of positions to plot
    - return_image: Return the image buffer instead of displaying it

    Returns
    -------
    - None | Image buffer if return_image is True
    """
    _, ax = plt.subplots()

    h_range = np.linspace(h-Δh/2, h+Δh/2, N, endpoint=True)

    # Plot UT trajectory
    for i, h in enumerate(h_range):
        p = project_position(r, h, l, δ)
        for j, pos in enumerate(p):
            ax.scatter(pos[0], pos[1], label=f"Telescope {j+1}" if i==len(h_range)-1 else None, color=f"C{j}", s=1+14*i/len(h_range))

    # Plot UT positions at T=0
    p = project_position(r, 0*u.hourangle, l, δ)
    for pos in p[1:]:
        ax.scatter(pos[0], pos[1], color="black", marker="+")

    ax.set_aspect("equal")
    ax.set_xlabel(f"x [{p[0].unit}]")
    ax.set_ylabel(f"y [{p[0].unit}]")
    ax.set_title(f"Projected telescope positions over the time (8h long)")
    plt.legend()

    if return_image:
        buffer = BytesIO()
        plt.savefig(buffer,format='png')
        plt.close()
        return buffer.getvalue()
    plt.show()

# Interactive plot ------------------------------------------------------------

def iplot_positions(
        r: u.Quantity,
        h: u.Quantity,
        Δh: u.quantity,
        l: u.Quantity,
        δ: u.Quantity,
    ):
    
    # GUI elements
    l_slider = widgets.FloatSlider(
        value=l.to(u.deg).value,
        min=-90,
        max=90,
        step=0.01,
        description='Latitude:',
    )
    δ_slider = widgets.FloatSlider(
        value=δ.to(u.deg).value,
        min=-90,
        max=90,
        step=0.01,
        description='Declination:',
    )
    reset = widgets.Button(description="Reset to defalut")
    plot = widgets.Image(width=500,height=500)

    # Callbacks
    def update_plot(*_):
        plot.value = plot_positions(
            r=r,
            h=h,
            Δh=Δh,
            l=l_slider.value*u.deg,
            δ=δ_slider.value*u.deg,
            N=11,
            return_image=True)

    def reset_values(*_):
        l_slider.value = l.to(u.deg).value
        δ_slider.value = δ.to(u.deg).value

    # Triggers
    reset.on_click(reset_values)
    l_slider.observe(update_plot, 'value')
    δ_slider.observe(update_plot, 'value')

    # Display
    display(widgets.VBox([l_slider, δ_slider, reset, plot]))
    update_plot()
    return