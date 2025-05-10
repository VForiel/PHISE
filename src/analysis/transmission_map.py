from .. import Scene

def plot(context, n:int=100):
    """
    Plot the transmission maps of the scene.
    
    Parameters
    ----------
    - context: Scene object
    - n: Resolution of the map
    """

    return context.plot_transmission_maps(N=n)