from ..classes.context import Context

def plot(context, n:int=100):
    """
    Plot the transmission maps of the context.
    
    Parameters
    ----------
    - context: Context object
    - n: Resolution of the map
    """

    return context.plot_transmission_maps(N=n)