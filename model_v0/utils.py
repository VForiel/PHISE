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










def optimize(kn0, beam, verbose = False):
    """------------------------------------------------------------------------
    Dichotomically optimize the phase shifters to maximize the bright output and minimize the
    dark pairs' difference.

    Parameters
    ----------
    - `kn0` : An instance of the KernelNuller class.
    - `beam` : The input beam complex amplitudes.
    - `verbose` : A boolean, whether to print the optimization steps.

    Returns
    -------
    - The optimized phase shifters
    - The evolution of the bright output during the optimization
    - The evolution of the dark pairs' difference during the optimization
    - The evolution of the phase shifters during the optimization
    ------------------------------------------------------------------------"""

    treshold = 1e-10

    # Shifters that contribute to redirecting light to the bright output
    p1 = [2,3,4,5,7] # 1,
    p1 = [1,2,3,4,5,7]

    # Shifters that contribute to the symmetry of the dark outputs
    p2 = [6,8,11,13,14] # 9,10,12,
    p2 = [6,8,11,13,14,9,10,12]


    # Initial phase shifters set to 0
    shifts = np.zeros(14)
    # Record the initial state
    shifts_evol = [np.copy(shifts)]

    #==========================================================================
    # PHASE 1: Maximise the bright output
    #==========================================================================
    if verbose and False: print("+"*10, " Phase 1 ", "+"*10)

    delta = 1
    bright_evol = []
    while delta > treshold:
        if verbose and False: print("\n==========")

        # Reduce the step size
        delta /= 2

        for i in p1:
            i = i-1 # convert human ndex to computer index

            change = np.zeros(len(shifts))
            change[i] = delta

            if verbose and False: 
                print("---")
                print(f"Initial shift_powers : {shifts}")
                print(f"Delta : {delta:.2e}, applied on P{i+1}")

            old_bright,_,_ = kn0(beam, shifts)
            pos_bright,_,_ = kn0(beam, shifts + change)
            neg_bright,_,_ = kn0(beam, shifts - change)

            # Metric: |bright|^2
            old_bright = np.abs(old_bright)**2
            pos_bright = np.abs(pos_bright)**2
            neg_bright = np.abs(neg_bright)**2

            if verbose and False :
                print(
                    f"Old bright : {old_bright:.2e},",
                    f"Pos bright : {pos_bright:.2e},",
                    f"Neg bright : {neg_bright:.2e}")

            # If negative shift is better, we apply it
            if  pos_bright > old_bright and pos_bright > old_bright:
                if verbose and False: print(f"-> Pos ({pos_bright:.2e})")
                shifts += change
                bright_evol.append(pos_bright)

            # If positive shift is better, we apply it
            elif neg_bright > old_bright and neg_bright > old_bright:
                if verbose and False: print(f"-> Neg ({neg_bright:.2e})")
                shifts -= change
                bright_evol.append(neg_bright)

            # If old parameters are better, we do nothing
            else:
                if verbose and False: print(f"-> Old ({old_bright:.2e})")
                bright_evol.append(old_bright)

            shifts_evol.append(np.copy(shifts))

    input_shifters = np.copy(shifts[:4])

    #==========================================================================
    # PHASE 2: Optimize dark pairs symmetry
    #==========================================================================
    if verbose: print("+"*10, " Phase 2 ", "+"*10)

    delta = 1
    dark_symmetry_evol = []
    
    # Perturb input phases
    kn0.noise_input_shifters()

    while delta > treshold:
        if verbose: print("\n==========\n")
        
        # Reduce the step size
        delta /= 2

        for i in p2:
            i = i-1 # convert human ndex to computer index

            change = np.zeros(len(shifts))
            change[i] = delta

            if verbose:
                print(f"\nShifts : {" ".join([f'{x:.2e}' for x in shifts])}")
                print(f"Delta : {delta:.2e}, applied on P{i+1}")

            _,old_darks,_ = kn0(beam, shifts)
            _,pos_darks,_ = kn0(beam, shifts + change)
            _,neg_darks,_ = kn0(beam, shifts - change)

            # Metric: |I1 - I2| + |I3 - I4| + |I5 - I6|
            # With In the intensity of the dark output n
            old_metric = np.abs(
                np.abs(old_darks[0])**2 - np.abs(old_darks[1])**2 +
                np.abs(old_darks[2])**2 - np.abs(old_darks[3])**2 +
                np.abs(old_darks[4])**2 - np.abs(old_darks[5])**2
            )
            pos_metric = np.abs(
                np.abs(pos_darks[0])**2 - np.abs(pos_darks[1])**2 +
                np.abs(pos_darks[2])**2 - np.abs(pos_darks[3])**2 +
                np.abs(pos_darks[4])**2 - np.abs(pos_darks[5])**2
            )
            neg_metric = np.abs(
                np.abs(neg_darks[0])**2 - np.abs(neg_darks[1])**2 +
                np.abs(neg_darks[2])**2 - np.abs(neg_darks[3])**2 +
                np.abs(neg_darks[4])**2 - np.abs(neg_darks[5])**2
            )

            if verbose :
                print(
                    f"Old metric : {old_metric:.2e},",
                    f"Pos metric : {pos_metric:.2e},",
                    f"Neg metric : {neg_metric:.2e}")

            # If negative shift is better, we apply it
            if  pos_metric < old_metric and pos_metric < old_metric:
                if verbose: print(f"-> Pos ({pos_metric:.2e})")
                shifts += change
                dark_symmetry_evol.append(pos_metric)

            # If positive shift is better, we apply it
            elif neg_metric < old_metric and neg_metric < old_metric:
                if verbose: print(f"-> Neg ({neg_metric:.2e})")
                shifts -= change
                dark_symmetry_evol.append(neg_metric)

            # If old parameters are better, we do nothing
            else:
                if verbose: print(f"-> Old ({old_metric:.2e})")
                dark_symmetry_evol.append(old_metric)

            if verbose and len(dark_symmetry_evol) > 2 and dark_symmetry_evol[-1] < dark_symmetry_evol[-2]:
                print("\n", "^"*30)

            shifts_evol.append(np.copy(shifts))

    shifts[:4] = input_shifters

    return shifts, bright_evol, dark_symmetry_evol, np.array(shifts_evol)










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
