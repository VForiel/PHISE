import os
import matplotlib.pyplot as plt
from pathlib import Path

def save_plot(save_as, default_name, fig=None):
    """
    Save a plot to a file.
    
    Parameters
    ----------
    save_as : str
        The path or directory to save the plot to.
        If it generally looks like a file (has extension), it is used as is.
        Otherwise, it is treated as a directory and default_name is appended.
    default_name : str
        The default filename to use if save_as is a directory.
    fig : matplotlib.figure.Figure, optional
        The figure to save. If None, plt.gcf() is used.
    """
    if not save_as:
        return

    path = Path(save_as)
    
    # Check if save_as has extension
    if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']:
        # It's a file path
        pass
    else:
        # It's a directory
        path = path / default_name
        
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if fig:
        fig.savefig(path, bbox_inches='tight')
    else:
        plt.savefig(path, bbox_inches='tight')
