import time
import numpy as np
import matplotlib.pyplot as plt

def optimize(kn0, beam, verbose = False):

    shifts = np.zeros(14)
    shifts_evol = [np.copy(shifts)]

    # Phase 1: redirect light to the bright output
    if verbose: print("+"*10, " Phase 1 ", "+"*10)
    delta = 1
    bright_evol = []
    while delta > 1e-20:
        delta /= 1.5
        if verbose: print("\n==========\n")
        p = [1,2,3,4,5,7] # shifters that contribute to redirecting light to the bright output
        for i in p:
            i = i-1 # convert human ndex to computer index

            change = np.zeros(len(shifts))
            change[i] = delta

            if verbose: 
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

            if verbose : print(f"Old bright : {old_bright:.2e}, Pos bright : {pos_bright:.2e}, Neg bright : {neg_bright:.2e}")

            # If negative shift is better, we apply it
            if  pos_bright > old_bright and pos_bright > old_bright:
                if verbose: print(f"-> Pos ({pos_bright:.2e})")
                shifts += change
                bright_evol.append(pos_bright)

            # If positive shift is better, we apply it
            elif neg_bright > old_bright and neg_bright > old_bright:
                if verbose: print(f"-> Neg ({neg_bright:.2e})")
                shifts -= change
                bright_evol.append(neg_bright)

            # If old parameters are better, we do nothing
            else:
                if verbose: print(f"-> Old ({old_bright:.2e})")
                bright_evol.append(old_bright)

            shifts_evol.append(np.copy(shifts))

    # Phase 2: optimize dark pairs symmetry
    if verbose: print("+"*10, " Phase 2 ", "+"*10)
    delta = 1
    dark_symmetry_evol = []
    while delta > 1e-20:
        delta /= 1.5
        if verbose: print("\n==========\n")
        p = [6,8,9,10,11,12,13,14] # shifters that contribute to the symmetry of the dark outputs
        for i in p:
            i = i-1 # convert human ndex to computer index

            change = np.zeros(len(shifts))
            change[i] = delta

            if verbose: 
                print("---")
                print(f"Initial shift_powers : {shifts}")
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

            if verbose : print(f"Old metric : {old_metric:.2e}, Pos metric : {pos_metric:.2e}, Neg metric : {neg_metric:.2e}")

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

            shifts_evol.append(np.copy(shifts))

    return shifts, bright_evol, dark_symmetry_evol, np.array(shifts_evol)

def bruteforce():

    step = 0.2
    a = -1
    b = 1
    N = int((b-a)/step)
    n = 14
    p = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    for x in range(N**n):

        for i in range(14):
            p[i] = a + (b-a) * (x // N**i % N) / N
            

def scan(kn, beams, scan_on=(1,2), initial_parameters=None, optimized_parameters=None, plot_intermediate_states=False):
    """------------------------------------------------------------------------
    Scan the parameter space and plot the null depths for each parameter
    combination.
    
    Parameters
    ----------
    kn : An instance of the KernelNuller class.
    beams : A list of 2D arrays, each representing a beam.
    optimized_parameters : A list of 14 floats, the optimized parameters.
    plot_intermediate_states : A boolean, whether to plot the intermediate"""

    # Scan shift power parameter space
    scan = np.linspace(-1, 1, 101, endpoint=True)
    null_depths = np.zeros((6, len(scan), len(scan)))
    brights = np.zeros((len(scan), len(scan)))
    first_layer_nulls = np.zeros((4, len(scan), len(scan)))
    second_layer_nulls = np.zeros((4, len(scan), len(scan)))

    if plot_intermediate_states:
        fig_b, axs_b = plt.subplots(2,4, figsize=(20,10))

    fig, axs = plt.subplots(3,3, figsize=(20,15))

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
                null_depths[k,i,j] = np.abs(dark)**2 # np.log(np.abs(dark)**2 / np.abs(bright)**2)
                brights[i,j] = np.abs(bright)**2

    if plot_intermediate_states:
        for k in range(4):
            p = axs_b[0, k]
            p.set_title(f"N{k//2} - {k%2}")
            im = p.imshow(first_layer_nulls[k])
            p.set_xlabel(f"Parameter {scan_on[1]}")
            p.set_ylabel(f"Parameter {scan_on[0]}")
            plt.colorbar(im)

        for k in range(4):
            p = axs_b[1, k]
            p.set_title(f"N{k//2+2} - {k%2}")
            im = p.imshow(second_layer_nulls[k])
            p.set_xlabel(f"Parameter {scan_on[1]}")
            p.set_ylabel(f"Parameter {scan_on[0]}")
            plt.colorbar(im)

    for k in range(6):
        p = axs[k%2, k//2]
        p.set_title(f"Dark {k+1}")
        im = p.imshow(null_depths[k], extent=[-1, 1, -1, 1])
        if optimized_parameters is not None: p.scatter(optimized_parameters[scan_on[1]-1], optimized_parameters[scan_on[0]-1], color='red')
        p.set_xlabel(f"Parameter {scan_on[1]}")
        p.set_ylabel(f"Parameter {scan_on[0]}")
        plt.colorbar(im)

    p = axs[2,0]
    p.set_title(f"Mean of null depths")
    im = p.imshow(np.mean(null_depths, axis=0), extent=[-1, 1, -1, 1])
    if optimized_parameters is not None: p.scatter(optimized_parameters[scan_on[1]-1], optimized_parameters[scan_on[0]-1], color='red')
    p.set_xlabel(f"Parameter {scan_on[1]}")
    p.set_ylabel(f"Parameter {scan_on[0]}")
    plt.colorbar(im)

    p = axs[2,2]
    p.set_title(f"Std of null depths")
    im = p.imshow(np.std(null_depths, axis=0), extent=[-1, 1, -1, 1])
    if optimized_parameters is not None: p.scatter(optimized_parameters[scan_on[1]-1], optimized_parameters[scan_on[0]-1], color='red')
    p.set_xlabel(f"Parameter {scan_on[1]}")
    p.set_ylabel(f"Parameter {scan_on[0]}")
    plt.colorbar(im)

    p = axs[2,1]
    p.set_title(f"Bright")
    im = p.imshow(brights, extent=[-1, 1, -1, 1])
    if optimized_parameters is not None: p.scatter(optimized_parameters[scan_on[1]-1], optimized_parameters[scan_on[0]-1], color='red')
    p.set_xlabel(f"Parameter {scan_on[1]}")
    p.set_ylabel(f"Parameter {scan_on[0]}")
    plt.colorbar(im)
    
    plt.show()
