import time
import numpy as np
import matplotlib.pyplot as plt

def optimize(kn0, beam, verbose = False):

    delta = 1
    null_depth_evol = []

    shifts = np.zeros(14)

    while delta > 1e-20:
        delta /= 1.5

        if verbose: print("\n==========\n")

        for i in range(len(shifts)):

            if verbose: 
                print("---")
                print(f"Initial shift_powers : {shifts}")
                print(f"Delta : {delta:.2e}, applied on P{i+1}")

            old_bright, old_darks, _ = kn0(beam)

            change = np.zeros(len(shifts))
            change[i] = delta
            
            pos_bright, pos_darks, _ = kn0(beam, shifts + change)
            neg_bright, neg_darks, _ = kn0(beam, shifts - change)

            old_null_mean = np.sum(np.abs(old_darks)) / len(old_darks) / np.abs(old_bright)
            pos_null_mean = np.sum(np.abs(pos_darks)) / len(pos_darks) / np.abs(pos_bright)
            neg_null_mean = np.sum(np.abs(neg_darks)) / len(neg_darks) / np.abs(neg_bright)

            null_depth_evol.append(old_null_mean)

            if verbose: print(f"Old ({old_null_mean:.2e}), Pos ({pos_null_mean:.2e}), Neg ({neg_null_mean:.2e})")
            if pos_null_mean < old_null_mean and pos_null_mean < neg_null_mean:
                if verbose: print(f"-> Pos ({pos_null_mean:.2e})")
                shifts += change
            elif neg_null_mean < old_null_mean and neg_null_mean < pos_null_mean:
                if verbose: print(f"-> Neg ({neg_null_mean:.2e})")
                shifts -= change
            else:
                if verbose: print(f"-> Old ({old_null_mean:.2e})")

    return null_depth_evol, shifts

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
    
    scan_on = (1,2) # Select 2 values from 1 to 14

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
