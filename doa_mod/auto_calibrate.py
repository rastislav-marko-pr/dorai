"""auto_calibrate — estimate mic array geometry from audio, no manual input.

Walk around the array while speaking for ~10 seconds. The module measures
pairwise TDOAs across many source positions and reconstructs the 2D mic
layout via multidimensional scaling (MDS).

Algorithm
---------
1. Capture overlapping frames while the user moves around the array.
2. For each frame, compute all M*(M-1)/2 pairwise GCC-PHAT TDOAs.
3. For each mic pair (i,j), the largest observed |τ_ij| over diverse source
   positions approximates d_ij / c  (the endfire TDOA = inter-mic distance
   divided by the speed of sound).
4. Classical MDS on the resulting distance matrix yields 2D relative
   positions, determined up to rotation and reflection.
5. A subsequent reference-direction step (optional) fixes the rotation.

Accuracy depends on the user traversing enough angle around the array. A
full semicircle is sufficient; a narrow arc underestimates some distances.
The module warns when distance estimates look unreliable.
"""
from __future__ import annotations

import time

import numpy as np

import doa_core as dc


# ---------------------------------------------------------------------------
# Pairwise distance estimation
# ---------------------------------------------------------------------------

def estimate_pairwise_distances(
    blocks: list[np.ndarray],
    fs: int,
    f_lo: float = dc.F_LO,
    f_hi: float = dc.F_HI,
    percentile: float = 95.0,
) -> tuple[np.ndarray, dict]:
    """Estimate inter-mic distances from a collection of multichannel blocks.

    Each block is [n_mics, n_samples]. Returns a symmetric distance matrix
    in metres and a dict of diagnostic info.

    Uses the percentile (default 95th) of |TDOA| across frames rather than
    the absolute maximum, which is more robust to outlier frames caused by
    reflections or transient noise.
    """
    n_mics = blocks[0].shape[0]
    pairs = dc.pairs_for(n_mics)
    n_pairs = len(pairs)

    all_taus = np.empty((len(blocks), n_pairs))
    all_peaks = np.empty((len(blocks), n_pairs))
    for f, block in enumerate(blocks):
        for k, (i, j) in enumerate(pairs):
            all_taus[f, k], all_peaks[f, k] = dc.gcc_phat(
                block[i], block[j], fs=fs, f_lo=f_lo, f_hi=f_hi)

    abs_taus = np.abs(all_taus)
    max_taus = np.percentile(abs_taus, percentile, axis=0)
    distances = max_taus * dc.C_SOUND

    # Symmetric distance matrix
    D = np.zeros((n_mics, n_mics))
    for k, (i, j) in enumerate(pairs):
        D[i, j] = distances[k]
        D[j, i] = distances[k]

    median_taus = np.median(abs_taus, axis=0)
    # spread >> 1 means diverse source positions were observed
    spread = max_taus / np.maximum(median_taus, 1e-12)

    info = {
        'distances_m': distances,
        'max_taus_us': max_taus * 1e6,
        'median_taus_us': median_taus * 1e6,
        'spread': spread,
        'n_frames': len(blocks),
        'pairs': pairs,
        'all_taus': all_taus,
    }
    return D, info


# ---------------------------------------------------------------------------
# Classical MDS
# ---------------------------------------------------------------------------

def mds_2d(D: np.ndarray) -> np.ndarray:
    """Classical multidimensional scaling: distance matrix → 2D positions.

    Returns [n_mics, 2] positions in metres, centred at the origin.
    The solution is determined up to an arbitrary rotation and reflection —
    use rotate_to_reference() to fix those if you have a known bearing.
    """
    n = D.shape[0]
    D2 = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n          # centering matrix
    B = -0.5 * H @ D2 @ H                        # double centering

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:2]           # two largest
    vals = np.maximum(eigvals[idx], 0.0)          # clamp noise
    pos = eigvecs[:, idx] * np.sqrt(vals)[None, :]
    pos -= pos.mean(axis=0)
    return pos


# ---------------------------------------------------------------------------
# Geometry classification
# ---------------------------------------------------------------------------

def is_linear(pos: np.ndarray, tol_ratio: float = 0.08) -> bool:
    """True if the estimated positions are approximately collinear.

    Checks whether the second MDS coordinate is small compared to the first
    (i.e. the array is basically one-dimensional). `tol_ratio` is the
    threshold on  spread_y / spread_x  below which the array is classified
    as linear.
    """
    if pos.shape[0] < 3:
        return True
    sx = np.max(pos[:, 0]) - np.min(pos[:, 0])
    sy = np.max(pos[:, 1]) - np.min(pos[:, 1])
    if sx < 1e-9:
        return sy < 1e-9       # all mics at the origin — degenerate
    return (sy / sx) < tol_ratio


def linear_gaps_from_positions(pos: np.ndarray) -> np.ndarray:
    """Extract inter-mic gaps (in metres) from a linear position array.

    Sorts mics along the dominant axis and returns the M-1 consecutive gaps.
    """
    # Project onto the principal axis
    c = pos.mean(axis=0)
    centred = pos - c
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    proj = centred @ Vt[0]
    order = np.argsort(proj)
    sorted_proj = proj[order]
    gaps = np.diff(sorted_proj)
    return gaps, order


# ---------------------------------------------------------------------------
# Reference-direction rotation
# ---------------------------------------------------------------------------

def rotate_to_reference(pos: np.ndarray, ref_block: np.ndarray,
                        ref_az_deg: float, fs: int,
                        f_hi: float = dc.F_HI) -> np.ndarray:
    """Rotate calibrated positions so a known source reads ref_az_deg.

    Record a short block with a source at a KNOWN bearing, pass it here, and
    the coordinate system is fixed. Without this, 0° is arbitrary but every
    bearing is self-consistent.
    """
    if is_linear(pos):
        # For linear arrays, compute the bearing the current geometry gives
        gaps, order = linear_gaps_from_positions(pos)
        # Reorder block to match sorted positions
        ordered_block = ref_block[order]
        est = dc.doa_linear(ordered_block, gaps, fs=fs, f_hi=f_hi)
        current_az = est['az_deg']
    else:
        est = dc.doa_least_squares(ref_block, pos=pos, fs=fs, f_hi=f_hi)
        current_az = est['az_deg']

    delta = np.deg2rad(ref_az_deg - current_az)
    R = np.array([[np.cos(delta), -np.sin(delta)],
                  [np.sin(delta),  np.cos(delta)]])
    return (R @ pos.T).T


# ---------------------------------------------------------------------------
# Main calibration entry point
# ---------------------------------------------------------------------------

def auto_calibrate(
    source,
    seconds: float = 10.0,
    block_size: int = 2048,
    hop: int = 512,
    vad_threshold: float = 0.45,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Run auto-calibration: capture audio while user moves, estimate geometry.

    Parameters
    ----------
    source : MicSource, WavSource, or MultiDeviceSource
        Already-started audio source with .read(n) returning [n_mics, n].
    seconds : float
        How long to capture for calibration.
    block_size : int
        Analysis frame length in samples.
    hop : int
        Hop between read() calls (sleep interval).
    vad_threshold : float
        Neural VAD speech probability threshold (0–1). Frames with max
        probability above this are considered speech-active.
    verbose : bool
        Print progress and diagnostics.

    Returns
    -------
    pos : ndarray [n_mics, 2]
        Estimated 2D mic positions in metres.
    info : dict
        Diagnostic information including distance estimates, linearity, etc.
    """
    fs = int(getattr(source, 'fs', 16000))

    # Load Neural VAD
    try:
        from vad import NeuralVAD
        vad = NeuralVAD()
        use_neural_vad = True
        if verbose:
            print("[calibrate] using Neural VAD for speech detection")
    except Exception as exc:
        vad = None
        use_neural_vad = False
        if verbose:
            print(f"[calibrate] Neural VAD not available ({exc}), "
                  f"falling back to RMS energy gating")

    if verbose:
        print("[calibrate] capturing audio — walk around the array while speaking…")

    blocks = []
    speech_flags = []
    t0 = time.time()
    sleep_s = hop / fs * 0.8

    while time.time() - t0 < seconds:
        block = source.read(block_size)
        if block is None:
            time.sleep(sleep_s)
            continue
        blocks.append(block.copy())

        if use_neural_vad:
            # Run VAD on the first mic channel (reference mic)
            result = vad.process_buffer(block[0], fs=fs,
                                        threshold=vad_threshold)
            speech_flags.append(result['is_speech'])
        else:
            speech_flags.append(dc.frame_rms(block) > 0.0)  # placeholder

        time.sleep(sleep_s)

    if len(blocks) < 20:
        raise RuntimeError(
            f"only {len(blocks)} frames captured in {seconds:.0f}s — "
            f"need at least 20. Check that the audio source is running "
            f"and the mics are connected.")

    # VAD: keep speech-active frames only
    if not use_neural_vad:
        # Fallback: RMS percentile gating
        rms_vals = [dc.frame_rms(b) for b in blocks]
        thr = np.percentile(rms_vals, 55.0)
        speech_flags = [r > thr for r in rms_vals]

    active_blocks = [b for b, is_speech in zip(blocks, speech_flags) if is_speech]

    if verbose:
        print(f"[calibrate] {len(active_blocks)} speech-active frames "
              f"out of {len(blocks)} total")

    if len(active_blocks) < 10:
        raise RuntimeError(
            f"only {len(active_blocks)} speech-active frames — speak louder "
            f"or move closer during calibration.")

    n_mics = active_blocks[0].shape[0]

    # Estimate pairwise distances
    # Use a generous f_hi for calibration since we don't know aperture yet
    D, dist_info = estimate_pairwise_distances(
        active_blocks, fs=fs, f_hi=dc.F_HI)

    # MDS → 2D positions
    pos = mds_2d(D)

    # Classify geometry
    linear = is_linear(pos)
    dist_info['is_linear'] = linear
    dist_info['n_mics'] = n_mics

    # Compute aperture and proper f_hi from the estimated geometry
    pairs = dc.pairs_for(n_mics)
    aperture = max(np.linalg.norm(pos[i] - pos[j]) for i, j in pairs)
    f_hi = min(dc.F_HI, dc.C_SOUND / (2.0 * aperture))
    dist_info['aperture_m'] = aperture
    dist_info['f_hi'] = f_hi

    if verbose:
        print(f"[calibrate] estimated {n_mics}-mic "
              f"{'LINEAR' if linear else 'PLANAR'} array")
        print(f"[calibrate] aperture {aperture*1000:.0f} mm, "
              f"band {dc.F_LO:.0f}–{f_hi:.0f} Hz")
        for k, (i, j) in enumerate(pairs):
            d = dist_info['distances_m'][k]
            print(f"[calibrate]   mic{i+1}↔mic{j+1}: {d*1000:.1f} mm "
                  f"(τ_max {dist_info['max_taus_us'][k]:.0f} µs, "
                  f"spread {dist_info['spread'][k]:.1f}×)")

        # Warn about low angular diversity
        low = [k for k in range(len(pairs)) if dist_info['spread'][k] < 1.5]
        if low:
            names = [f"mic{pairs[k][0]+1}↔mic{pairs[k][1]+1}" for k in low]
            print(f"[calibrate] [!] low angular diversity for {', '.join(names)}")
            print(f"[calibrate]     walk a wider arc to improve the estimate")

        if linear:
            gaps, order = linear_gaps_from_positions(pos)
            print(f"[calibrate] channel order along the line: "
                  f"{[f'mic{o+1}' for o in order]}")
            print(f"[calibrate] gaps: {np.round(gaps*1000, 1)} mm")
            print("[calibrate] NOTE: a linear array CANNOT resolve which side "
                  "of the line a source is on.")
            print("[calibrate]       Both mirror branches will be shown.")

    return pos, dist_info
