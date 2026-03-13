@nb.njit()
def get_output_fields_batch_jit(
        ψ: np.ndarray[complex],
        φ: np.ndarray[float],
        σ: np.ndarray[float],
        λ: float,
        λ0: float,
        output_order: np.ndarray[int]
    ) -> np.ndarray[complex]:
    """Simulate a 4-telescope Kernel Nuller propagation (batch mode).
    
    Args:
        ψ (np.ndarray[complex]): Input fields (N_batch, 4) or (4,).
        φ (np.ndarray[float]): OPDs (N_batch, 14).
        σ (np.ndarray[float]): Intrinsic errors (14,) or (N_batch, 14).
        ...
        
    Returns:
        np.ndarray[complex]: Output fields (N_batch, 7).
    """
    # Create local variables to avoid reshaping overhead inside loops if possible
    # But for batching, we just operate on arrays.
    
    n_batch = φ.shape[0]
    λ_ratio = λ0 / λ
    
    # Precompute matrices (same as single version)
    N = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    Na, Nφ = np.abs(N), np.angle(N)
    N = Na * np.exp(1j * Nφ * λ_ratio)
    
    θ = np.pi / 2
    R = 1 / np.sqrt(2) * np.array([
        [np.exp(1j * θ / 2), np.exp(-1j * θ / 2)],
        [np.exp(-1j * θ / 2), np.exp(1j * θ / 2)]
    ], dtype=np.complex128)
    Ra, Rφ = np.abs(R), np.angle(R)
    R = Ra * np.exp(1j * Rφ * λ_ratio)
    
    # Prepare inputs
    # Broadcast ψ if necessary
    if ψ.ndim == 1:
        # (4,) -> (1, 4)
        ψ_batch = ψ.reshape(1, 4)
        # We can rely on broadcasting `ψ_batch` * `exp(...)` if sizes align
        # But `phase.shift_jit` expects matching dims or scalar.
        # Let's handle broadcasting explicitly or assume `ψ` matches `φ`'s batch dim if needed
        # In our case, ψ is constant for the whole batch of scans for ONE sample.
        # So ψ (4,) against φ (280, 14)
        # We need ψ to be (280, 4) effectively for the first operation?
        # Actually `shift_jit` does `ψ * exp(...)`.
        pass

    # Better approach might be to just loop over the batch in Numba (parallel=True maybe?)
    # Or write vectorized operations.
    
    # Vectorized implementation:
    
    # Total effective phase
    Φ = φ + σ # (N, 14)
    
    # 1. Input shift
    # ψ: (4,) complex. Φ[:, :4]: (N, 4). 
    # shift_jit handles broadcasting if we pass ψ (4,) and δφ (N, 4)?
    # Let's check shift_jit again.
    # return ψ * np.exp(1j * 2 * np.pi * δφ / λ)
    # (4,) * (N, 4) -> (N, 4) via numpy broadcasting rules?
    # Yes, (4,) broadcasts to (N, 4) if last dim matches.
    
    ψ0 = phase.shift_jit(ψ, Φ[:, :4], λ) # -> (N, 4)
    
    # 2. First Nuller Layer
    # We need to apply N (2, 2) to pairs of columns.
    # ψ0 shape (N, 4).
    # Pair 1: ψ0[:, :2] (N, 2). Pair 2: ψ0[:, 2:] (N, 2).
    # Matmul: (2, 2) @ (N, 2).T -> (2, N) -> .T -> (N, 2)
    # Or just manual:
    # out0 = N[0,0]*in0 + N[0,1]*in1
    # out1 = N[1,0]*in0 + N[1,1]*in1
    
    # Let's manually unroll for speed and clarity
    # N is constant.
    
    # ... (implementation details)
    return ...
