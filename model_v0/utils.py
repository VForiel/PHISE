import time
import numpy as np
import matplotlib.pyplot as plt





def bound_phase(phase:np.array) -> np.array:
    """------------------------------------------------------------------------
    Bound the phase between [-1,1] (normalized from [-pi,pi])

    Parameters
    ----------
    - `phase` : Normalized input phase that can be out of bounds [-1,1]

    Returns
    -------
    - Normalized phase bounded in [-1,1]
    ------------------------------------------------------------------------"""
    return (phase + 1) % 2 - 1










def phase_shift(beam:complex, dphase:float) -> complex:
    """------------------------------------------------------------------------
    De-phase the input beam by heating the fiber with an electrical current.
    
    Parameters
    ----------
    - `beam` : Input beam complex amplitude
    - `shift` : Normalized phase shift ([-1,1] ->[-pi,pi])
    
    Returns
    -------
    - Output beam complex amplitude
    ------------------------------------------------------------------------"""
    return beam * np.exp(1j * dphase * np.pi)










def mach_zehnder(beam:complex, dphase:float) -> complex:
    """------------------------------------------------------------------------
    Partially or totally cancel the light of a beam by
    splitting it in two part, dephasing one path and recombining them.
    
    Parameters
    ----------
    - `input_beam` : Input beam complex amplitude
    - `input_power` : Input power in Watts for the dephasing.
    
    Returns
    -------
    - Output beam complex amplitude
    ------------------------------------------------------------------------"""
    return (beam + phase_shift(beam, dphase)) / 2










def nuller4x4(beams:list[complex]) -> tuple[complex, list[complex]]:
    """------------------------------------------------------------------------
    Simulate a 4 input beam nuller.
    
    Parameters
    ----------
    - `beams` : List of 4 input beams complex amplitudes
    
    Returns
    -------
    - Bright channel complex amplitude
    - List of 3 dark channels complex amplitudes
    ------------------------------------------------------------------------"""

    N = 1/np.sqrt(4) * np.array([
        [1,  1,  1,  1],
        [1,  1, -1, -1],
        [1, -1,  1, -1],
        [1, -1, -1,  1]
    ])
    
    outputs = N @ beams

    return outputs[0], outputs[1:]










def nuller_2x2(beams:np.array) -> np.array:
    """------------------------------------------------------------------------
    Simulate a 2 input beam nuller.

    Parameters
    ----------
    - `beams` : Array of 2 input beams complex amplitudes

    Returns
    -------
    - Array of 2 output beams complex amplitudes
        - 1st output is the bright channel
        - 2nd output is the dark channel
    ------------------------------------------------------------------------"""

    N = 1/np.sqrt(2) * np.array([
        [1,   1],
        [1,  -1],
    ])

    return N @ beams










def splitmix_4x4(beams:list[complex]) -> list[complex]:
    """------------------------------------------------------------------------
    Simulate a 3 input beam split and mix.
    
    Parameters
    ----------
    - `beams` : List of input beams complex amplitudes 
    
    Returns
    -------
    - List of output beams complex amplitudes
        --------------------------------------------------------------------"""
    
    phi = np.pi/4
    S = 1/np.sqrt(4) * np.array([
        [1               , np.exp(1j*phi)  , 0             ],
        [-np.exp(-1j*phi), 1               , 0             ],
        [1               , 0               , np.exp(1j*phi)],
        [-np.exp(-1j*phi), 0               , 1             ],
        [0               , 1               , np.exp(1j*phi)],
        [0               , -np.exp(-1j*phi), 1             ]
    ])

    return S @ beams










def splitmix_2x2(beams:np.array, theta=np.pi/2) -> np.array:
    """------------------------------------------------------------------------
    Simulate a 2 input beam split and mix.

    Parameters
    ----------
    - `beams` : Array of 2 input beams complex amplitudes
    - `theta` : Phase shift between the two output beams

    Returns
    -------
    - Array of 2 output beams complex amplitudes
    ------------------------------------------------------------------------"""

    S = 1/np.sqrt(2) * np.array([
        [np.exp(1j*theta/2), np.exp(-1j*theta/2)],
        [np.exp(-1j*theta/2), np.exp(1j*theta/2)]
    ])

    return S @ beams










def bruteforce():
    """------------------------------------------------------------------------
    Brute force the parameter space to find the optimal parameters.

    # TODO
    ------------------------------------------------------------------------"""

    step = 0.2
    a = -1
    b = 1
    N = int((b-a)/step)
    n = 14
    p = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    for x in range(N**n):

        for i in range(14):
            p[i] = a + (b-a) * (x // N**i % N) / N










def scan(kn, beams, scan_on=(1,2), initial_parameters=None,
         optimized_parameters=None, plot_intermediate_states=False):
    """------------------------------------------------------------------------
    Scan the parameter space and plot the null depths for each parameter
    combination.
    
    Parameters
    ----------
    kn : An instance of the KernelNuller class.
    beams : A list of 2D arrays, each representing a beam.
    optimized_parameters : A list of 14 floats, the optimized parameters.
    plot_intermediate_states : A boolean, whether to plot the intermediate

    Returns
    -------
    None
    ------------------------------------------------------------------------"""

    # Scan shift power parameter space
    scan = np.linspace(-1, 1, 101, endpoint=True)
    darks_map = np.zeros((6, len(scan), len(scan)))
    bright_map = np.zeros((len(scan), len(scan)))

    if plot_intermediate_states:
        first_layer_nulls = np.zeros((4, len(scan), len(scan)))
        second_layer_nulls = np.zeros((4, len(scan), len(scan)))

        _, axs_b = plt.subplots(2,4, figsize=(20,10))
    _, axs = plt.subplots(4,3, figsize=(20,25))

    if initial_parameters is not None:
        parameters = np.array(initial_parameters).astype(float)
    else:
        parameters = np.copy(kn.shift_powers)

    for i, scan1 in enumerate(scan):
        for j, scan2 in enumerate(scan):
            parameters[scan_on[0]-1] = scan1
            parameters[scan_on[1]-1] = scan2

            # print("params=",parameters, scan1, scan2)

            bright, darks, inter = kn(beams, parameters)

            if plot_intermediate_states:
                for k, null in enumerate(inter['first_nuller_layer']):
                    first_layer_nulls[k,i,j] = np.abs(null)**2

                for k, null in enumerate(inter['second_nuller_layer']):
                    second_layer_nulls[k,i,j] = np.abs(null)**2

            for k, dark in enumerate(darks):
                darks_map[k,i,j] = np.abs(dark)**2
                bright_map[i,j] = np.abs(bright)**2

    if plot_intermediate_states:
        mi = min(np.min(first_layer_nulls), np.min(second_layer_nulls))
        ma = max(np.max(first_layer_nulls), np.max(second_layer_nulls))

        for k in range(4):
            p = axs_b[0, k]
            p.set_title(f"N{k//2+1} - {k%2}")
            im = p.imshow(first_layer_nulls[k], vmin=mi, vmax=ma)
            p.set_xlabel(f"Parameter {scan_on[1]}")
            p.set_ylabel(f"Parameter {scan_on[0]}")
            plt.colorbar(im)

        for k in range(4):
            p = axs_b[1, k]
            p.set_title(f"N{k//2+3} - {k%2}")
            im = p.imshow(second_layer_nulls[k], vmin=mi, vmax=ma)
            p.set_xlabel(f"Parameter {scan_on[1]}")
            p.set_ylabel(f"Parameter {scan_on[0]}")
            plt.colorbar(im)

    for k in range(6):
        p = axs[k%2, k//2]
        p.set_title(f"Dark {k+1}")
        im = p.imshow(darks_map[k], extent=[-1, 1, -1, 1], vmin=np.min(darks_map), vmax=np.max(darks_map))
        if optimized_parameters is not None: p.scatter(optimized_parameters[scan_on[1]-1], optimized_parameters[scan_on[0]-1], color='red')
        p.set_xlabel(f"Parameter {scan_on[1]}")
        p.set_ylabel(f"Parameter {scan_on[0]}")
        plt.colorbar(im)

    # Plot diff of dark pairs -------------------------------------------------
        
    diff1 = darks_map[0] - darks_map[1]
    diff2 = darks_map[2] - darks_map[3]
    diff3 = darks_map[4] - darks_map[5]

    # Compute max diff
    m = np.max([np.max(np.abs(diff1)), np.max(np.abs(diff2)), np.max(np.abs(diff3))])

    p = axs[2,0]
    p.set_title(f"Dark 1 - Dark 2")
    im = p.imshow(diff1, extent=[-1, 1, -1, 1], cmap='RdBu', vmin=-m, vmax=m)
    if optimized_parameters is not None: p.scatter(optimized_parameters[scan_on[1]-1], optimized_parameters[scan_on[0]-1], color='red')
    p.set_xlabel(f"Parameter {scan_on[1]}")
    p.set_ylabel(f"Parameter {scan_on[0]}")
    plt.colorbar(im)

    p = axs[2,1]
    p.set_title(f"Dark 3 - Dark 4")
    im = p.imshow(diff2, extent=[-1, 1, -1, 1], cmap='RdBu', vmin=-m, vmax=m)
    if optimized_parameters is not None: p.scatter(optimized_parameters[scan_on[1]-1], optimized_parameters[scan_on[0]-1], color='red')
    p.set_xlabel(f"Parameter {scan_on[1]}")
    p.set_ylabel(f"Parameter {scan_on[0]}")
    plt.colorbar(im)

    p = axs[2,2]
    p.set_title(f"Dark 5 - Dark 6")
    im = p.imshow(diff3, extent=[-1, 1, -1, 1], cmap='RdBu', vmin=-m, vmax=m)
    if optimized_parameters is not None: p.scatter(optimized_parameters[scan_on[1]-1], optimized_parameters[scan_on[0]-1], color='red')
    p.set_xlabel(f"Parameter {scan_on[1]}")
    p.set_ylabel(f"Parameter {scan_on[0]}")
    plt.colorbar(im)

    p = axs[3,0]
    p.set_title(f"Mean of null depths")
    im = p.imshow(np.mean(darks_map, axis=0), extent=[-1, 1, -1, 1])
    if optimized_parameters is not None: p.scatter(optimized_parameters[scan_on[1]-1], optimized_parameters[scan_on[0]-1], color='red')
    p.set_xlabel(f"Parameter {scan_on[1]}")
    p.set_ylabel(f"Parameter {scan_on[0]}")
    plt.colorbar(im)

    p = axs[3,2]
    p.set_title(f"Std of null depths")
    im = p.imshow(np.std(darks_map, axis=0), extent=[-1, 1, -1, 1])
    if optimized_parameters is not None: p.scatter(optimized_parameters[scan_on[1]-1], optimized_parameters[scan_on[0]-1], color='red')
    p.set_xlabel(f"Parameter {scan_on[1]}")
    p.set_ylabel(f"Parameter {scan_on[0]}")
    plt.colorbar(im)

    p = axs[3,1]
    p.set_title(f"Bright")
    im = p.imshow(bright_map, extent=[-1, 1, -1, 1])
    if optimized_parameters is not None: p.scatter(optimized_parameters[scan_on[1]-1], optimized_parameters[scan_on[0]-1], color='red')
    p.set_xlabel(f"Parameter {scan_on[1]}")
    p.set_ylabel(f"Parameter {scan_on[0]}")
    plt.colorbar(im)
    
    plt.show()
