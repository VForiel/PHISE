import numpy as np
import astropy.units as u
from LRFutils import color
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from copy import deepcopy as copy

from src.modules import signals
from src.modules import phase
from src.classes import kernel_nuller
from src.classes.kernel_nuller import KernelNuller

#==============================================================================
# Deterministic genetic algorithm
#==============================================================================

def genetic(
        kn: KernelNuller,
        β: float,
        λ: u.Quantity,
        f: u.Quantity,
        Δt: u.Quantity,
        verbose: bool = False,
        plot=False,
    ) -> tuple[u.Quantity, dict[str, np.ndarray[float]]]:
    """
    Optimize the phase shifters offsets to maximize the nulling performance

    Parameters
    ----------
    - kn: Kernel nuller object
    - β: Decay factor for the step size (0.5 <= β < 1)
    - λ: Wavelength of the observation
    - f: Flux of the star
    - Δt: Integration time
    - verbose: Boolean, if True, print the optimization process

    Returns
    -------
    - Dict containing the history of the optimization
    """

    ψ = np.ones(4) * (1 + 0j) * np.sqrt(f.to(1/Δt.unit).value/4) # Perfectly cophased inputs

    ε = 1e-6 * λ.unit # Minimum shift step size

    # Shifters that contribute to redirecting light to the bright output
    φb = [1, 2, 3, 4, 5, 7]

    # Shifters that contribute to the symmetry of the dark outputs
    φk = [6, 8, 9, 10, 11, 12, 13, 14]

    # History of the optimization
    bright_history = []
    kernel_history = []
    shifters_history = []

    Δφ = λ / 4
    while Δφ > ε:

        if verbose:
            print(color.black(color.on_red(f"--- New iteration ---")), f"Δφ={Δφ:.2e}")

        for i in φb + φk:
            log = ""

            # Step vector
            s = np.zeros(14) * λ.unit
            s[i-1] = Δφ

            # Apply the step
            _, k_neg, b_neg = kernel_nuller.observe_njit(ψ=ψ, φ=kn.φ-s, σ=kn.σ, λ=λ, Δt=Δt.value)
            _, k_old, b_old = kernel_nuller.observe_njit(ψ=ψ, φ=kn.φ,   σ=kn.σ, λ=λ, Δt=Δt.value)
            _, k_pos, b_pos = kernel_nuller.observe_njit(ψ=ψ, φ=kn.φ+s, σ=kn.σ, λ=λ, Δt=Δt.value)

            # Total Kernels relative intensity
            k_neg = np.sum(np.abs(k_neg))
            k_old = np.sum(np.abs(k_old))
            k_pos = np.sum(np.abs(k_pos))

            # Save the history
            bright_history.append(b_old / f.to(1/Δt.unit).value)
            kernel_history.append(k_old / f.to(1/Δt.unit).value)
            shifters_history.append(np.copy(kn.φ))

            # Maximize the bright metric for group 1 shifters
            if i in φb:
                log += "Shift " + color.black(color.on_lightgrey(f"{i}")) + " Bright: " + color.black(color.on_green(f"{b_neg:.2e} | {b_old:.2e} | {b_pos:.2e}")) + " -> "

                if b_pos > b_old and b_pos > b_neg:
                    log += color.black(color.on_green(" + "))
                    kn.φ += s
                elif b_neg > b_old and b_neg > b_pos:
                    log += color.black(color.on_green(" - "))
                    kn.φ -= s
                else:
                    log += color.black(color.on_green(" = "))

            # Minimize the kernel metric for group 2 shifters
            else:
                log += "Shift " + color.black(color.on_lightgrey(f"{i}")) + " Kernel: " + color.black(color.on_blue(f"{k_neg:.2e} | {k_old:.2e} | {k_pos:.2e}")) + " -> "

                if k_pos < k_old and k_pos < k_neg:
                    kn.φ += s
                    log += color.black(color.on_blue(" + "))
                elif k_neg < k_old and k_neg < k_pos:
                    kn.φ -= s
                    log += color.black(color.on_blue(" - "))
                else:
                    log += color.black(color.on_blue(" = "))
            
            if verbose:
                print(log)

        Δφ *= β

    kn.φ = phase.bound(kn.φ, λ)

    if plot:
        plot_genetic_history({
            "bright": np.array(bright_history),
            "kernel": np.array(kernel_history),
            "shifters": np.array(shifters_history),
        })

def plot_genetic_history(history: dict[str, np.ndarray[float]]):
    bright_evol = history["bright"]
    kernel_evol = history["kernel"]
    shifts_evol = history["shifters"]

    _, axs = plt.subplots(3,1, figsize=(15,15))

    axs[0].plot(bright_evol)
    axs[0].set_xlabel("Number of iterations")
    axs[0].set_ylabel("Bright throughput (%)")
    axs[0].set_yscale("log")
    axs[0].set_title("Optimization of the bright output")

    axs[1].plot(kernel_evol)
    axs[1].set_xlabel("Number of iterations")
    axs[1].set_ylabel("Kernels throughput (%)")
    axs[1].set_yscale("log")
    axs[1].set_title("Optimization of the kernels")

    print(np.mean(kernel_evol[-2000:]))

    for i in range(shifts_evol.shape[1]):
        axs[2].plot(shifts_evol[:,i], label=f"Shifter {i+1}")
    axs[2].set_xlabel("Number of iterations")
    axs[2].set_ylabel("Phase shift")
    axs[2].set_yscale("linear")
    axs[2].set_title("Convergeance of the phase shifters")
    axs[2].legend(loc='upper right')

    plt.show()

#==============================================================================
# Input obstruction algorithm
#==============================================================================

def obstruction(
        kn: KernelNuller,
        λ: u.Quantity,
        f: u.Quantity,
        Δt: u.Quantity,
        N = 1_000,
        plot: bool = False,
    ) -> tuple[u.Quantity, dict[str, np.ndarray[float]]]:
    """
    Optimize the phase shifters offsets to maximize the nulling performance

    Parameters
    ----------
    - kn: KernelNuller object
    - λ: Wavelength of the observation
    - f: Flux of the star
    - Δt: Integration time
    - N: Number of points for the least squares optimization
    - plot: Boolean, if True, plot the optimization process
    """

    def maximize_bright(kn:KernelNuller, ψ, n):

        x = np.linspace(0,λ.value,N)
        y = np.empty(N)

        for i in range(N):
            kn.φ[n-1] = i * λ / N
            _, _, b = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            y[i] = b / f.to(1/Δt.unit).value

        def sin(x, x0):
            return 1/4 * (np.sin((x-x0)/λ.value*2*np.pi)+1)/2
        popt, pcov = curve_fit(sin, x, y, p0=[0])

        kn.φ[n-1] = (np.mod(popt[0]+λ.value/4, λ.value) * λ.unit).to(kn.φ.unit)

        if plot:
            plt.scatter(x, y, label='Data', color='tab:blue')
            plt.plot(x, sin(x, *popt), label='Fit', color='tab:orange')
            plt.axvline(x=np.mod(popt[0]+λ.value/4, λ.value), color='k', linestyle='--', label='Optimal phase shift')
            plt.xlabel(f"Phase shift ({λ.unit})")
            plt.ylabel("Bright throughput")
            plt.title(f"Bright($\phi_{n}$)")
            plt.legend()
            plt.show()

    def minimize_kernel(kn, ψ, n, i):

        x = np.linspace(0,λ.value,N)
        y = np.empty(N)

        for i in range(N):
            kn.φ[n-1] = i * λ / N
            _, k, _ = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            y[i] = np.sum(np.abs(k)) / f.to(1/Δt.unit).value

        def sin(x, x0):
            return 1/8 * np.abs(np.sin((x-x0)/λ.value*2*np.pi))
        popt, pcov = curve_fit(sin, x, y, p0=[0])

        kn.φ[n-1] = (np.mod(popt[0], λ.value) * λ.unit).to(kn.φ.unit)

        if plot:
            plt.scatter(x, y, label='Data', color='tab:blue')
            plt.plot(x, sin(x, *popt), label='Fit', color='tab:orange')
            plt.axvline(x=np.mod(popt[0], λ.value), color='k', linestyle='--', label='Optimal phase shift')
            plt.xlabel(f"Phase shift ({λ.unit})")
            plt.ylabel("Kernel throughput")
            plt.title(f"Kernels($\phi_{n}$)")
            plt.legend()
            plt.show()

    def maximize_darks(kn, ψ, n, ds):

        x = np.linspace(0,λ.value,N)
        y = np.empty(N)
    
        for i in range(N):
            kn.φ[n-1] = i * λ / N
            d, _, _ = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            y[i] = np.sum(np.abs(d[np.array(ds)-1])) / f.to(1/Δt.unit).value

        def sin(x, x0):
            return 1/4 * (np.sin((x-x0)/λ.value*2*np.pi)+1)/2
        popt, pcov = curve_fit(sin, x, y, p0=[0], maxfev = 100_000)

        kn.φ[n-1] = (np.mod(popt[0]+λ.value/4, λ.value) * λ.unit).to(kn.φ.unit)

        if plot:
            plt.scatter(x, y, label='Data', color='tab:blue')
            plt.plot(x, sin(x, *popt), label='Fit', color='tab:orange')
            plt.axvline(x=np.mod(popt[0]+λ.value/4, λ.value), color='k', linestyle='--', label='Optimal phase shift')
            plt.xlabel(f"Phase shift ({λ.unit})")
            plt.ylabel(f"Dark pair {ds} throughput")
            plt.title(f"Darks{ds}($\phi_{n}$)")
            plt.legend()
            plt.show()

    a = (1+0j) * np.sqrt(f) * np.sqrt(1/4)

    # Bright maximization
    ψ = np.array([a, a, 0, 0])
    maximize_bright(kn, ψ, 2)

    ψ = np.array([0, 0, a, a])
    maximize_bright(kn, ψ, 4)

    ψ = np.array([a, 0, a, 0])
    maximize_bright(kn, ψ, 7)

    # Darks maximization
    ψ = np.array([a, 0, -a, 0])
    maximize_darks(kn, ψ, 8, [1,2])

    # Kernel minimization
    ψ = np.array([a, 0, 0, 0])
    minimize_kernel(kn, ψ, 11, 1)
    minimize_kernel(kn, ψ, 13, 2)
    minimize_kernel(kn, ψ, 14, 3)

    kn.φ = phase.bound(kn.φ, λ)

#--------------------------------------------------------------------------------

def obstruction2(
        kn: KernelNuller,
        λ: u.Quantity,
        f: u.Quantity,
        Δt: u.Quantity,
        N = 1_000,
        plot: bool = False,
    ) -> tuple[u.Quantity, dict[str, np.ndarray[float]]]:
    """
    Optimize the phase shifters offsets to maximize the nulling performance

    Parameters
    ----------
    - kn: KernelNuller object
    - λ: Wavelength of the observation
    - f: Flux of the star
    - Δt: Integration time
    - N: Number of points for the least squares optimization
    - plot: Boolean, if True, plot the optimization process
    """

    def maximize_bright(kn:KernelNuller, ψ, n):

        x = np.array([0]*N + [λ.value/4]*N + [λ.value/2]*N + [3*λ.value/4]*N)
        y = np.empty(4*N)

        for i in range(N):
            kn.φ[n-1] = x[i] * λ.unit
            _, _, b = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            y[i] = b / f.to(1/Δt.unit).value

        def sin(x, x0):
            return 1/4 * (np.sin((x-x0)/λ.value*2*np.pi)+1)/2
        popt, pcov = curve_fit(sin, x, y, p0=[0])

        kn.φ[n-1] = (np.mod(popt[0]+λ.value/4, λ.value) * λ.unit).to(kn.φ.unit)

        if plot:
            plt.scatter(x, y, label='Data')
            plt.plot(np.linspace(0, λ.value, N*4), sin(np.linspace(0, λ.value, N*4), *popt), label='Fit')
            plt.axvline(x=np.mod(popt[0]+λ.value/4, λ.value), color='k', linestyle='--', label='Optimal phase shift')
            plt.xlabel(f"Phase shift ({λ.unit})")
            plt.ylabel("Bright throughput")
            plt.title(f"Bright($\phi_{n}$)")
            plt.legend()
            plt.show()

    def minimize_kernel(kn, ψ, n, i):

        x = np.array([0]*N + [λ.value/4]*N + [λ.value/2]*N + [3*λ.value/4]*N)
        y = np.empty(4*N)

        for i in range(N):
            kn.φ[n-1] = x[i] * λ.unit
            _, k, _ = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            y[i] = np.sum(np.abs(k)) / f.to(1/Δt.unit).value

        def sin(x, x0):
            return 1/8 * np.abs(np.sin((x-x0)/λ.value*2*np.pi))
        popt, pcov = curve_fit(sin, x, y, p0=[0])

        kn.φ[n-1] = (np.mod(popt[0], λ.value) * λ.unit).to(kn.φ.unit)

        if plot:
            plt.scatter(x, y, label='Data')
            plt.plot(np.linspace(0, λ.value, N*4), sin(np.linspace(0, λ.value, N*4), *popt), label='Fit')
            plt.axvline(x=np.mod(popt[0], λ.value), color='k', linestyle='--', label='Optimal phase shift')
            plt.xlabel(f"Phase shift ({λ.unit})")
            plt.ylabel("Kernel throughput")
            plt.title(f"Kernels($\phi_{n}$)")
            plt.legend()
            plt.show()

    def maximize_darks(kn, ψ, n, ds):

        x = np.array([0]*N + [λ.value/4]*N + [λ.value/2]*N + [3*λ.value/4]*N)
        y = np.empty(4*N)
    
        for i in range(N):
            kn.φ[n-1] = x[i] * λ.unit
            d, _, _ = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            y[i] = np.sum(np.abs(d[np.array(ds)-1])) / f.to(1/Δt.unit).value

        def sin(x, x0):
            return 1/4 * (np.sin((x-x0)/λ.value*2*np.pi)+1)/2
        popt, pcov = curve_fit(sin, x, y, p0=[0], maxfev = 100_000)

        kn.φ[n-1] = (np.mod(popt[0]+λ.value/4, λ.value) * λ.unit).to(kn.φ.unit)

        if plot:
            plt.scatter(x, y, label='Data')
            plt.plot(np.linspace(0, λ.value, N*4), sin(np.linspace(0, λ.value, N*4), *popt), label='Fit')
            plt.axvline(x=np.mod(popt[0]+λ.value/4, λ.value), color='k', linestyle='--', label='Optimal phase shift')
            plt.xlabel(f"Phase shift ({λ.unit})")
            plt.ylabel(f"Dark pair {ds} throughput")
            plt.title(f"Darks{ds}($\phi_{n}$)")
            plt.legend()
            plt.show()
            

    a = (1+0j) * np.sqrt(f) * np.sqrt(1/4)

    # Bright maximization
    ψ = np.array([a, a, 0, 0])
    maximize_bright(kn, ψ, 2)

    ψ = np.array([0, 0, a, a])
    maximize_bright(kn, ψ, 4)

    ψ = np.array([a, 0, a, 0])
    maximize_bright(kn, ψ, 7)

    # Darks maximization
    ψ = np.array([a, 0, -a, 0])
    maximize_darks(kn, ψ, 8, [1,2])

    # Kernel minimization
    ψ = np.array([a, 0, 0, 0])
    minimize_kernel(kn, ψ, 11, 1)
    minimize_kernel(kn, ψ, 13, 2)
    minimize_kernel(kn, ψ, 14, 3)

    kn.φ = phase.bound(kn.φ, λ)

#==============================================================================
# Comparison of the two algorithms
#==============================================================================

def compare_approaches(f:u.Quantity, Δt:u.Quantity, λ:u.Quantity):
    β_res = 10
    βs, dβ = np.linspace(0.5, 0.999, β_res, retstep=True)
    Ns = [10, 100, 1000, 10_000, 100_000]
    samples = 10

    # fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    shots = []
    for β in βs:
        for i in range(samples):
            print(f'Gen.: β={β:.3f}, sample={i+1}/{samples}          ', end='\r')
            kn = KernelNuller(φ=np.zeros(14)*λ, σ=np.random.uniform(0, 1, 14)*λ)
            history = genetic(kn=kn, β=β, λ=λ, f=f, Δt=Δt, verbose=False)
            ψ = np.ones(4) * (1+0j) * np.sqrt(1/4)
            _, d, b = kn.propagate_fields(ψ=ψ, λ=λ)
            di = np.abs(d)**2
            k = np.array([di[0] - di[1], di[2] - di[3], di[4] - di[5]])
            depth = np.sum(np.abs(k)) / np.abs(b)**2
            shots.append((len(history['bright']), depth))
        
    x, y = zip(*shots)
    x = np.array(x); y = np.array(y)
    plt.scatter(np.random.uniform(x-x/10, x+x/10), y, c='tab:blue', s=5, label='Genetic')

    slope, intercept, r_value, p_value, std_err = linregress(np.log10(x), np.log10(y))
    print(slope, intercept)
    x_values = np.linspace(min(x), max(x), 500)
    y_fit = 10**intercept * x_values**slope
    plt.plot(x_values, y_fit, 'tab:cyan', linestyle='--', label=f'Gen. fit')

    print("")

    shots = []
    for j, N in enumerate(Ns):
        for i in range(samples):
            print(f'Obs.: N={N}, sample={i+1}/{samples}          ', end='\r')
            kn = KernelNuller(φ=np.zeros(14)*λ, σ=np.random.uniform(0, 1, 14)*λ)
            obstruction(kn=kn, λ=λ, f=f, Δt=Δt, N=N, plot=False)
            ψ = np.ones(4) * (1+0j) * np.sqrt(1/4)
            _, d, b = kn.propagate_fields(ψ=ψ, λ=λ)
            di = np.abs(d)**2
            k = np.array([di[0] - di[1], di[2] - di[3], di[4] - di[5]])
            depth = np.sum(np.abs(k)) / np.abs(b)**2
            shots.append((7*N, depth))

    x, y = zip(*shots)
    x = np.array(x); y = np.array(y)
    plt.scatter(np.random.uniform(x-x/10, x+x/10), y, c='tab:orange', s=5, label='Obstruction')

    slope, intercept, r_value, p_value, std_err = linregress(np.log10(x), np.log10(y))
    print(slope, intercept)
    x_values = np.linspace(min(x), max(x), 500)
    y_fit = 10**intercept * x_values**slope
    plt.plot(x_values, y_fit, 'tab:red', linestyle='--', label=f'Obs. fit')

    plt.xlabel('# of iterations')
    plt.xscale('log')
    plt.ylabel('Depth')
    plt.yscale('log')
    plt.title('Efficiency of the calibration approaches')
    plt.legend()
    plt.show()

#==============================================================================
# Scan parameters
#==============================================================================

def scan(
    scan_on,
    kn: KernelNuller,
    λ: u.Quantity,
    restricted: bool = False,
):
    """
    Scan the parameter space and plot the null depths for each parameter
    combination.

    Parameters
    ----------
    - scan_on: List of two integers, the parameters to scan
    - kn: KernelNuller object
    - λ: Wavelength of the observation
    - restricted: Boolean, if True, consider only the errors and corrections

    Returns
    -------
    - None
    """
    kn = copy(kn) # ensure not editing a reference

    # Scan shift power parameter space
    scan = np.linspace(0, λ.value, 101, endpoint=True) * λ.unit

    # Initialize the maps
    nulls_map = np.zeros((3, len(scan), len(scan)))
    darks_map = np.zeros((6, len(scan), len(scan)))
    kernels_map = np.zeros((3, len(scan), len(scan)))
    bright_map = np.zeros((len(scan), len(scan)))

    # Create the figure
    _, axs = plt.subplots(3, 5, figsize=(30, 15))

    φ = kn.φ.copy()
    σ = kn.σ.copy()

    # Consider only errors & correction on the shifter that are being scanned
    if restricted:
        kn.φ = np.zeros(14) * λ.unit
        kn.σ = np.zeros(14) * λ.unit
        kn.σ[scan_on[0] - 1] = σ[scan_on[0] - 1]
        kn.σ[scan_on[1] - 1] = σ[scan_on[1] - 1]

    ψ = signals.get_input_fields(
        a=1,
        θ=0 * u.rad,
        α=0 * u.rad,
        λ=λ,
        p=np.array([[0, 0], [0, 0], [0, 0], [0, 0]]) * u.m,
    )

    for i, scan1 in enumerate(scan):
        for j, scan2 in enumerate(scan):
            kn.φ[scan_on[0] - 1] = scan1
            kn.φ[scan_on[1] - 1] = scan2

            nulls, darks, bright = kn.propagate_fields(ψ=ψ, λ=λ)
            
            kernels = np.array([
                    np.abs(darks[2*i])**2 - np.abs(darks[2*i+1])**2
                for i in range(3)])

            for k, null in enumerate(nulls):
                nulls_map[k, i, j] = np.abs(null)**2
            for k, dark in enumerate(darks):
                darks_map[k, i, j] = np.abs(dark)**2
            for k, kernel in enumerate(kernels):
                kernels_map[k, i, j] = kernel
            bright_map[i, j] = np.abs(bright)**2

    for k in range(3):
        p = axs[k, 0]
        p.set_title(f"Null {k+1}")
        im = p.imshow(
            nulls_map[k],
            extent=[0, λ.value, 0, λ.value],
            vmin=np.min(nulls_map),
            vmax=np.max(nulls_map),
        )
        p.scatter(
            φ[scan_on[1] - 1],
            φ[scan_on[0] - 1],
            color="red",
            edgecolors="white",
            s=100,
        )
        p.scatter(
            phase.bound(-σ, λ)[scan_on[1] - 1],
            phase.bound(-σ, λ)[scan_on[0] - 1],
            color="green",
            edgecolors="white",
            s=100,
        )
        p.set_xlabel(f"Parameter {scan_on[1]}")
        p.set_ylabel(f"Parameter {scan_on[0]}")
        plt.colorbar(im)

    for k in range(6):
        p = axs[k // 2, k % 2 + 1]
        p.set_title(f"Dark {k+1}")
        im = p.imshow(
            darks_map[k],
            extent=[0, λ.value, 0, λ.value],
            vmin=np.min(darks_map),
            vmax=np.max(darks_map),
            cmap="hot",
        )
        p.scatter(
            φ[scan_on[1] - 1],
            φ[scan_on[0] - 1],
            color="red",
            edgecolors="white",
            s=100,
        )
        p.scatter(
            phase.bound(-σ, λ)[scan_on[1] - 1],
            phase.bound(-σ, λ)[scan_on[0] - 1],
            color="green",
            edgecolors="white",
            s=100,
        )
        p.set_xlabel(f"Parameter {scan_on[1]}")
        p.set_ylabel(f"Parameter {scan_on[0]}")
        plt.colorbar(im)

    for k in range(3):
        p = axs[k, 3]
        p.set_title(f"Kernel {k+1}")
        im = p.imshow(
            kernels_map[k],
            extent=[0, λ.value, 0, λ.value],
            vmin=np.min(kernels_map),
            vmax=np.max(kernels_map),
            cmap="bwr",
        )
        p.scatter(
            φ[scan_on[1] - 1],
            φ[scan_on[0] - 1],
            color="red",
            edgecolors="white",
            s=100,
        )
        p.scatter(
            phase.bound(-σ, λ)[scan_on[1] - 1],
            phase.bound(-σ, λ)[scan_on[0] - 1],
            color="green",
            edgecolors="white",
            s=100,
        )
        p.set_xlabel(f"Parameter {scan_on[1]}")
        p.set_ylabel(f"Parameter {scan_on[0]}")
        plt.colorbar(im)

    p = axs[1, 4]
    p.set_title(f"Bright")
    im = p.imshow(bright_map, extent=[0, λ.value, 0, λ.value], cmap="gray")
    p.scatter(
        φ[scan_on[1] - 1],
        φ[scan_on[0] - 1],
        color="red",
        edgecolors="white",
        s=100,
    )
    p.scatter(
        phase.bound(-σ, λ)[scan_on[1] - 1],
        phase.bound(-σ, λ)[scan_on[0] - 1],
        color="green",
        edgecolors="white",
        s=100,
    )
    p.set_xlabel(f"Parameter {scan_on[1]}")
    p.set_ylabel(f"Parameter {scan_on[0]}")
    plt.colorbar(im)

    axs[0, 4].axis("off")
    axs[2, 4].axis("off")

    plt.show()