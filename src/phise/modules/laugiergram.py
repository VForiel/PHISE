import numpy as np
import matplotlib.pyplot as plt

def laugiergram(phasors, ax=None, marker='*', labels=None, colors=None):
    """
    Plots a Laugiergram on a given matplotlib polar axis.
    Displays lines from the center (origin) to points representing
    the complex numbers in polar coordinates. The angle is the argument
    (phase) and the radius is the norm (amplitude) of each complex number.
    
    Parameters
    ----------
    phasors : array-like
        An array-like of complex numbers to plot.
    ax : matplotlib.axes.Axes, optional
        A matplotlib polar axis instance where the Laugiergram will be plotted.
    marker : str, default='*'
        The marker style to use at the end of the line.
    labels : list of str, optional
        A list of string labels of the same length as phasors.
    colors : list of colors, optional
        A list of colors of the same length as phasors. If None,
        matplotlib's default color cycle is used to assign a unique color per phasor.

    Returns
    -------
    ax : axes or tuple
        If ax was provided, returns the ax. 
        If no ax was provided, returns a tuple (fig, ax).
    """
    return_fig = False
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        return_fig = True

    z = np.asarray(phasors, dtype=complex)
    
    theta = np.angle(z)
    r = np.abs(z)
    
    if labels is not None:
        if len(labels) != len(z):
            raise ValueError(f"Length of labels ({len(labels)}) must match length of phasors ({len(z)})")
            
    if colors is not None:
        if len(colors) != len(z):
            raise ValueError(f"Length of colors ({len(colors)}) must match length of phasors ({len(z)})")
        color_list = colors
    else:
        # Generate unique colors for each phasor using the default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        color_list = [c['color'] for _, c in zip(range(len(z)), prop_cycle)]
        # If there are more phasors than colors in the cycle, it wraps around automatically 
        # because zip stops, so we need to use itertools.cycle or similar.
        # However, a simpler way is to just index into a repeating list:
        cycle_colors = prop_cycle.by_key()['color']
        color_list = [cycle_colors[i % len(cycle_colors)] for i in range(len(z))]
    
    # We use a loop to plot each line from the origin to the point (th, rad)
    # in polar coordinates. The angle th is repeated so it draws a straight radial line.
    for i, (th, rad) in enumerate(zip(theta, r)):
        color = color_list[i]
        # Line from origin
        ax.plot([th, th], [0, rad], color=color, label=labels[i] if labels is not None else None)
        # Point at the end with marker and black edge
        ax.plot([th], [rad], marker=marker, markersize=12, color=color,
                markeredgecolor='black', markeredgewidth=0.5)
    
    if labels is not None:
        ax.legend()
            
    if return_fig:
        return fig, ax
    return ax
