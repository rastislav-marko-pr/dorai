"""doa_core — direction-of-arrival estimation for the ReSpeaker 4-Mic array.

Two estimators, both phase-transform based so they are robust to the level
differences between mics on this board:

  gcc_phat(a, b)        pairwise TDOA, sub-sample, via GCC-PHAT
  doa_least_squares()   6 pairwise TDOAs -> one azimuth by least squares
  srp_phat()            steered-response-power spatial spectrum over azimuth

Both were validated on a real 10.2 s ReSpeaker capture: they agreed to 1.0 deg
(+89.0 vs +88.0) with circular sd 5.4 deg over speech-active frames.

BAND LIMIT IS LOAD-BEARING
--------------------------
The array's 64.0 mm diagonal aliases spatially above c/(2*d) = 2680 Hz (the
45.3 mm adjacent spacing aliases above 3790 Hz). Measured per-band on the real
capture, the 2000-2600 Hz band already degrades to concentration R = 0.652
while its neighbours hold R >= 0.99. F_HI is therefore 2600 Hz, NOT Nyquist.
Raising it does not add resolution, it adds ambiguous peaks.

GEOMETRY
--------
Channel order was derived from the data, not from a datasheet: the six pairwise
TDOAs satisfy plane-wave closure to <= 1.45 mm, and a square with channels
wrapping sequentially around the circle fits with 2.00 mm residual against
3.34 mm for the alternating arrangement. Nominal radius 32 mm is the published
ODAS/pyroomacoustics figure; the plane-wave constraint |u| = 1 on the real
capture implies 35.7 mm (+11.6%). AZIMUTH IS INVARIANT to this radius scaling
(it is a ratio of direction-vector components), so the nominal value is kept as
the default. ELEVATION IS NOT invariant -- treat any elevation derived from
|u| as unreliable until the true radius is measured.
"""
from __future__ import annotations

import numpy as np

C_SOUND = 343.0          # m/s
FS_DEFAULT = 16000
F_LO = 200.0             # Hz — below this, room modes and handling noise dominate
F_HI = 2600.0            # Hz — spatial aliasing limit of the 64 mm diagonal
R_NOMINAL = 0.032        # m — circle radius; see GEOMETRY note above
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))   # the 4-mic default


def pairs_for(n_mics):
    """All unordered channel pairs for an n-mic array, in ascending order.

    n_mics=4 reproduces the validated PAIRS tuple exactly.
    """
    return tuple((i, j) for i in range(n_mics) for j in range(i + 1, n_mics))


def array_positions(radius=R_NOMINAL):
    """[4,2] mic xy positions in metres, channel order ch1..ch4.

    ReSpeaker 4-Mic Array: MIC1 at 45 deg, numbering COUNTER-CLOCKWISE, i.e.
    45/135/225/315 deg. This matches the Seeed board drawing (DOA:0 right,
    DOA:90 top) and is identical to circle_positions(4, radius, start_deg=45).

    HOW THIS WAS PINNED, and why the previous value was wrong
    ---------------------------------------------------------
    The earlier value was [135, 45, -45, -135], which is this layout MIRRORED
    about the x-axis -- not rotated. A mirror cannot be absorbed as a constant
    offset, so it cannot be calibrated away.

    Scored against the four marked talker positions of the 2026-08-04 Panza
    session (0/90/180/270 deg at 1.5 m), brute-forcing all 24 channel->capsule
    permutations x reflection on noise-subtracted SRP spectra:

        this layout           7.5 deg mean abs error
        runner-up            20.2 deg
        previous [135,45,..] 87.0 deg

    Confirmed on the two-talker takes, which took no part in the selection.

    Why it went unnoticed for so long: with the gcc_phat sign also inverted
    (see the SIGN CONVENTION note there), a reflection LOOKS like a rotation on
    any source lying near the mirror axis -- and the 0/180 deg marks lie
    exactly on it. A rotation-only fit therefore reported a clean "+178 deg
    offset, no mirror" and fit all four marks about equally badly.

    NOTE both faults had to be fixed together. Either one alone makes accuracy
    WORSE than shipping neither (73 deg -> 107 deg sign-only / 132 deg
    geometry-only -> 48 deg both). tests/test_doa_core.py covers this.
    """
    ang = np.deg2rad([45.0, 135.0, 225.0, 315.0])
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)


def circle_positions(n_mics, radius, start_deg=90.0, clockwise=False,
                     centre_mic=False):
    """[N,2] mic xy positions on a uniform circular array (UCA), metres.

    `radius` is in METRES. Mics are evenly spaced by 360/n_mics degrees
    starting at `start_deg` (measured from +x, counter-clockwise by default).

    Two physically different arrays get called "5-mic circle" in the wild:

      * a 5-element UCA -- five mics on the rim, 72 deg apart:
            circle_positions(5, 0.0463)
      * four on the rim plus ONE AT THE CENTRE (common on dev boards):
            circle_positions(4, 0.0463, centre_mic=True)

    They have different aperture and different resolution, and the centre mic
    contributes no delay of its own -- it only adds pairs. Passing the wrong
    one produces a confident wrong bearing, so check the board before choosing.

    `start_deg`/`clockwise` fix which physical mic is channel 1 and which way
    the numbering runs. Both are conventions of the board's wiring, not of the
    geometry, and getting them wrong ROTATES or MIRRORS every bearing. If the
    reported angle is consistently offset from truth, that is the first thing
    to check -- calibrate_offset() against a known direction will show it.
    """
    n = int(n_mics)
    if n < 2:
        raise ValueError(f"need at least 2 rim mics, got {n}")
    step = -360.0 / n if clockwise else 360.0 / n
    ang = np.deg2rad(start_deg + step * np.arange(n))
    rim = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    if centre_mic:
        return np.vstack([rim, np.zeros((1, 2))])
    return rim


def pair_positions(spacing):
    """[2,2] positions for TWO mics on the x-axis, separated by `spacing` metres.

    The baseline defines the coordinate system: mic1 at -spacing/2, mic2 at
    +spacing/2, so bearings are measured from the +x axis (mic2's side).

    A two-element array cannot resolve a full 360 deg azimuth. It measures only
    the angle to the BASELINE AXIS, so every estimate has a mirror twin
    reflected across that axis (+40 deg and -40 deg give identical delays).
    Use doa_pair(), which returns both branches explicitly, rather than
    doa_least_squares(), whose 2x2 normal equations are rank-1 here and would
    silently return a minimum-norm answer collapsed onto the axis.
    """
    h = 0.5 * float(spacing)
    return np.array([[-h, 0.0], [h, 0.0]])


def pair_baselines(pos, pairs=None):
    """[P,2] baseline vectors p_i - p_j, in `pairs` order."""
    if pairs is None:
        pairs = PAIRS if len(pos) == 4 else pairs_for(len(pos))
    return np.stack([pos[i] - pos[j] for i, j in pairs])


def gcc_phat(a, b, fs=FS_DEFAULT, f_lo=F_LO, f_hi=F_HI, max_tau=3e-4, up=32):
    """Sub-sample TDOA of b relative to a. Positive tau => b lags a.

    The phase transform (dividing by |R|) whitens the cross-spectrum, so the
    peak location depends on phase alone. That is what makes this work across
    mics with different sensitivities.

    max_tau bounds the search to physically reachable delays; for a 64 mm
    aperture the true bound is 64mm/c = 187 us, and 300 us leaves margin.
    Returns (tau_seconds, normalized_peak_height).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    nfft = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    R = A * np.conj(B)
    R = R / np.maximum(np.abs(R), 1e-12)          # phase transform
    R[(freqs < f_lo) | (freqs > f_hi)] = 0.0      # band limit — see module docstring
    cc = np.fft.irfft(R, nfft * up)
    lag_max = int(round(max_tau * fs * up))
    # SIGN CONVENTION -- read this before touching the two lines below.
    #
    # R = A*conj(B), so cc[n] is the correlation at lag +n IN THE SENSE
    # "a advanced by n", i.e. cc peaks at index +D when a(t) = b(t - D), which
    # is b LEADING a. The window is laid out as [negative lags | 0 | positive],
    # so index k maps to lag (lag_max - k) in the DOCUMENTED convention
    # (positive tau => b lags a), NOT (k - lag_max).
    #
    # The original code used (k - lag_max), which returned the exact negative
    # of the documented value. Verified against the 2026-08-04 Panza session:
    # feed this function a signal and an exact +100 us delayed copy and the old
    # form returned -100 us. Every bearing came out 180 deg wrong, and because
    # srp_phat carried the SAME inverted convention the two estimators agreed
    # with each other while both were wrong -- which is why the original
    # "they agree to 1.0 deg" validation did not catch it.
    #
    # tests/test_doa_core.py::test_gcc_phat_sign locks this down. Do not
    # "simplify" it back.
    cc = np.concatenate([cc[-lag_max:], cc[:lag_max + 1]])
    k = int(np.argmax(cc))
    # parabolic interpolation on the already-upsampled peak
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        denom = y0 - 2 * y1 + y2
        delta = (y0 - y2) / (2 * denom) if denom != 0 else 0.0
    else:
        delta = 0.0
    tau = ((lag_max - k) - delta) / (fs * up)
    # Peak-to-sidelobe ratio: the peak height against the typical level OUTSIDE
    # a guard band around it. A single dominant source gives a sharp, isolated
    # peak (high ratio); diffuse noise, reverberation, or two streams whose
    # clocks have walked apart give a low, ambiguous one.
    #
    # The previous definition, cc[k]/max(|cc|), was ~1.0 by construction
    # whenever k was the maximum -- it carried no information at all. This
    # matters most with TWO microphones, where there is only one pair and so no
    # residual redundancy to cross-check against: the sidelobe ratio is then
    # the ONLY signal that a frame is trustworthy.
    guard = max(1, int(0.15e-3 * fs * up))          # ±150 µs around the peak
    lo, hi = max(0, k - guard), min(len(cc), k + guard + 1)
    side = np.concatenate([cc[:lo], cc[hi:]])
    if side.size:
        floor = float(np.sqrt((side ** 2).mean()))
        peak = float(cc[k] / (floor + 1e-20))
    else:
        peak = 0.0
    return tau, peak


def doa_pair(a, b, spacing, fs=FS_DEFAULT, f_lo=F_LO, f_hi=None, tau_offset=0.0):
    """Bearing from ONE microphone pair -- returns BOTH mirror branches.

    A two-element array measures only the angle to its baseline axis. The
    mapping is tau = (spacing/c) * cos(theta), so theta and -theta are
    indistinguishable: a talker 40 deg above the baseline and one 40 deg below
    produce identical delays. This function returns theta_deg in [0,180] plus
    its mirror at -theta_deg, and never pretends to a unique answer.

    f_hi defaults to the pair's own spatial aliasing limit c/(2*spacing),
    capped at F_HI -- a wide pair aliases at a LOWER frequency than the
    ReSpeaker's 2600 Hz and must be band-limited accordingly.

    tau_offset (seconds) is subtracted before the geometry inversion: it is the
    fixed instrumental delay between the two capture paths, which is NOT zero
    for two independent USB devices. Get it from calibrate_offset().

    Returns dict(theta_deg, mirror_deg, tau, cos_arg, saturated, peak).
    `saturated` is True when |cos| had to be clipped, i.e. the measured delay
    exceeded what the geometry allows -- a sign of drift, a wrong `spacing`, or
    a reflection-dominated frame.
    """
    if f_hi is None:
        f_hi = min(F_HI, C_SOUND / (2.0 * float(spacing)))
    tau, peak = gcc_phat(a, b, fs=fs, f_lo=f_lo, f_hi=f_hi,
                         max_tau=max(1.5 * float(spacing) / C_SOUND, 1e-4))
    tau_c = tau - tau_offset
    cos_arg = tau_c * C_SOUND / float(spacing)
    saturated = bool(abs(cos_arg) > 1.0)
    theta = float(np.rad2deg(np.arccos(np.clip(cos_arg, -1.0, 1.0))))
    return dict(theta_deg=theta, mirror_deg=-theta, tau=tau, cos_arg=cos_arg,
                saturated=saturated, peak=peak)


def linear_positions(spacings, along='x'):
    """[M,2] positions for mics in a straight LINE, from gap distances.

    `spacings` is the gap between consecutive mics in metres: one number for a
    uniform array, or M-1 numbers for an irregular one. The array is centred on
    the origin and laid along +x (or +y), in channel order.

        linear_positions(0.14)          -> 3 mics? no: needs M, see below
        linear_positions([0.14, 0.14])  -> 3 mics at -0.14, 0, +0.14

    This is all the geometry a linear array needs: only distances ALONG the
    line affect the delays.
    """
    s = np.atleast_1d(np.asarray(spacings, dtype=np.float64))
    x = np.concatenate([[0.0], np.cumsum(s)])
    x = x - x.mean()
    z = np.zeros_like(x)
    return np.stack([x, z], axis=1) if along == 'x' else np.stack([z, x], axis=1)


def doa_linear(block, spacings, fs=FS_DEFAULT, f_lo=F_LO, f_hi=None,
               side=None, tau_offsets=None, psr_min=1.8):
    """Bearing from mics in a straight LINE. Returns BOTH mirror branches.

    A linear array measures only the angle to its own AXIS. The delay to mic m
    is tau_m = (x_m/c)*cos(theta), and cos(theta) == cos(-theta), so a source at
    +40 deg and one at -40 deg produce byte-identical delays. This is the "cone
    of confusion": the source lies on a cone around the array axis, which in the
    plane collapses to two mirror bearings. No algorithm removes it -- the
    information is not in the recording. A non-collinear array is the only fix.

    Three mics still beat two: with 2 unknowns' worth of redundancy the three
    pairwise delays over-determine one angle, so `resid_us` becomes a real
    consistency check (large = reverberation, a wrong spacing, or clock drift).

    Parameters
    ----------
    spacings : float or sequence
        Gaps between consecutive mics in METRES, in channel order.
    side : {'+', '-', None}
        Which side of the line the source is known to be on, if you know. This
        does NOT come from the audio -- it is YOUR prior, used only to pick
        which branch is reported as `az_deg`. None reports the +branch and
        leaves the ambiguity fully visible.
    f_hi : float or None
        None uses the array's own spatial aliasing limit c/(2*d_max), capped at
        F_HI. A wide array aliases at a LOWER frequency and must be limited.

    Returns dict(theta_deg, mirror_deg, az_deg, cos_arg, saturated, resid_us,
                 peak_min, taus).
    `theta_deg` in [0,180] is the angle from the +x end of the array;
    90 deg is broadside. `saturated` means the measured delay exceeded what the
    geometry allows -- a wrong spacing, clock drift, or a reflection-dominated
    frame.
    """
    block = np.asarray(block, dtype=np.float64)
    M = block.shape[0]
    s = np.atleast_1d(np.asarray(spacings, dtype=np.float64))
    if s.size == 1:
        s = np.repeat(s, M - 1)
    if s.size != M - 1:
        raise ValueError(f"{M} mics need {M-1} spacing(s), got {s.size}")
    x = np.concatenate([[0.0], np.cumsum(s)])
    x = x - x.mean()
    d_max = float(x.max() - x.min())
    if f_hi is None:
        f_hi = min(F_HI, C_SOUND / (2.0 * d_max))

    pairs = pairs_for(M)
    if tau_offsets is None:
        tau_offsets = np.zeros(len(pairs))
    tau_offsets = np.asarray(tau_offsets, dtype=np.float64)

    taus = np.empty(len(pairs))
    peaks = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        mt = max(1.5 * abs(x[i] - x[j]) / C_SOUND, 1e-4)
        taus[k], peaks[k] = gcc_phat(block[i], block[j], fs=fs, f_lo=f_lo,
                                     f_hi=f_hi, max_tau=mt)
    taus = taus - tau_offsets

    # Least squares on the ONE unknown, cos(theta):
    #   tau_ij = (x_i - x_j)/c * cos(theta)
    dx = np.array([x[i] - x[j] for i, j in pairs]) / C_SOUND
    denom = float(dx @ dx)
    cos_arg = float(dx @ taus / denom) if denom > 0 else 0.0
    resid_us = float(np.sqrt((((dx * cos_arg) - taus) ** 2).mean()) * 1e6)

    saturated = bool(abs(cos_arg) > 1.0)
    theta = float(np.rad2deg(np.arccos(np.clip(cos_arg, -1.0, 1.0))))
    mirror = -theta
    if side == '-':
        az = mirror % 360.0
    else:
        az = theta % 360.0

    # Physical consistency gate. The largest delay the geometry can produce is
    # d_max/c; a residual approaching that means the pairwise delays cannot all
    # come from one plane wave, so the frame carries no direction. Causes, in
    # order of likelihood: independent capture clocks drifting apart, a wrong
    # `spacings`, or a reflection-dominated frame.
    #
    # This matters because averaging such frames does NOT cancel their error --
    # it pulls the estimate toward broadside (90 deg), which then looks like a
    # confident measurement. Measured: with 200 ppm drift, true bearings of
    # 10/15/30 deg all reported 85-90 deg.
    max_delay_us = d_max / C_SOUND * 1e6
    # TWO gates, because they catch different failures and one of them stops
    # working when the array shrinks:
    #
    #  * residual  -- needs REDUNDANCY. With M>=3 mics the pairwise delays
    #    over-determine one angle, so disagreement is measurable. With M==2
    #    there is a single pair and one unknown, so the residual is 0 BY
    #    CONSTRUCTION and this test cannot fire at all.
    #  * peak-to-sidelobe -- works at any M. A frame dominated by one source
    #    has a sharp isolated correlation peak; drift, reverberation, or noise
    #    flatten it.
    #
    # Measured on a 2-mic array with 120 ppm drift and a true bearing of 45°,
    # the residual test passed 43% of frames and reported 97.5°. Clean frames
    # sit at PSR ~2.6 (p10 2.56) where drifting ones fall to p10 0.05, so the
    # PSR floor is what protects a two-microphone setup.
    peak_min = float(peaks.min())
    consistent = (bool(resid_us <= 0.25 * max_delay_us)
                  and not saturated
                  and peak_min >= psr_min)
    return dict(theta_deg=theta, mirror_deg=mirror, az_deg=az,
                cos_arg=cos_arg, saturated=saturated, resid_us=resid_us,
                resid_frac=float(resid_us / max_delay_us) if max_delay_us else 0.0,
                consistent=consistent, max_delay_us=float(max_delay_us),
                peak_min=peak_min, n_mics=int(M), taus=taus, f_hi=float(f_hi))


def closure_error_us(taus, pairs=None, n_mics=4):
    """RMS TDOA closure error in microseconds. GEOMETRY-FREE frame-quality test.

    For ANY single source, whatever the array looks like and wherever the mics
    actually are, the delays around every closed triangle must sum to zero:

        tau_ik - tau_ij - tau_jk == 0

    because tau_ij = t_j - t_i telescopes. This holds for a wrong `pos`, a wrong
    radius, a mirrored channel map, any elevation, near field or far field --
    it depends on nothing but the existence of ONE arrival time per microphone.

    That independence is what makes it the right gate. The plane-wave residual
    from doa_least_squares() mixes two different failures into one number: it is
    large when the FRAME is bad (reverberation, two talkers, noise) and equally
    large when the GEOMETRY is wrong, and you cannot tell which from the value.
    Closure only responds to the first, so a large closure error means the audio
    genuinely carries no single direction. Use both: closure to reject frames,
    residual to detect a bad `pos`.

    Measured on the 2026-08-04 Panza session (aperture 64 mm, max delay 187 us):
    the quiet 1.5 m takes ran at 17-66 us median closure with essentially no
    coherent frames, while the ~20 dB louder two-talker takes ran at 16-21 us.
    Synthetic plane waves give < 0.01 us.

    Needs M >= 3. With 2 mics there is one pair, no triangle, and no closure to
    test -- fall back to the peak-to-sidelobe ratio from gcc_phat().
    """
    taus = np.asarray(taus, dtype=np.float64)
    if pairs is None:
        pairs = PAIRS if n_mics == 4 else pairs_for(n_mics)
    idx = {p: k for k, p in enumerate(pairs)}
    n = max(max(p) for p in pairs) + 1
    if n < 3:
        return 0.0
    errs = [taus[idx[(i, k)]] - taus[idx[(i, j)]] - taus[idx[(j, k)]]
            for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    return float(np.sqrt(np.mean(np.square(errs))) * 1e6)


def doa_least_squares(block, pos=None, fs=FS_DEFAULT, f_lo=F_LO, f_hi=F_HI,
                      tau_offsets=None, closure_max_us=None):
    """Azimuth from the pairwise TDOAs of one [M, N] block (M >= 3).

    Solves D u = c*tau in the least-squares sense, where D holds the baseline
    vectors and u is the unit direction toward the source. Using all six pairs
    (rather than two) averages down individual TDOA errors and yields a
    residual that doubles as a confidence measure.

    Returns dict(az_deg, u_norm, resid_mm, peak_min, taus). u_norm is
    cos(elevation) and should be <= 1; values above 1 indicate a geometry-scale
    error, not a real elevation (see the module GEOMETRY note).
    """
    block = np.asarray(block, dtype=np.float64)
    if pos is None:
        pos = array_positions()
    pos = np.asarray(pos, dtype=np.float64)
    n_mics = pos.shape[0]
    if n_mics < 3:
        raise ValueError(
            "doa_least_squares needs >= 3 mics: with 2 the 2x2 normal "
            "equations are rank-1 (all baselines are collinear) and the "
            "solution collapses onto the baseline axis. Use doa_pair().")
    pairs = PAIRS if n_mics == 4 else pairs_for(n_mics)
    D = pair_baselines(pos, pairs)
    # Collinear arrays are rank-deficient in 2-D for the same reason a pair is.
    if np.linalg.matrix_rank(D, tol=1e-9) < 2:
        raise ValueError(
            "microphone positions are collinear: a straight-line array cannot "
            "resolve which side of its axis a source is on. Offset one mic "
            "perpendicular to the line, or use doa_pair() per adjacent pair.")
    if tau_offsets is None:
        tau_offsets = np.zeros(len(pairs))
    tau_offsets = np.asarray(tau_offsets, dtype=np.float64)
    taus = np.empty(len(pairs))
    peaks = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        taus[k], peaks[k] = gcc_phat(block[i], block[j], fs=fs, f_lo=f_lo, f_hi=f_hi)
    d = (taus - tau_offsets) * C_SOUND
    u, *_ = np.linalg.lstsq(D, d, rcond=None)
    resid_mm = float(np.sqrt(((D @ u - d) ** 2).mean()) * 1000.0)
    # Geometry-free frame quality. Default threshold is 15% of the largest
    # delay the array can produce -- on the ReSpeaker that is ~28 us.
    d_max = max(np.linalg.norm(pos[i] - pos[j]) for i, j in pairs)
    max_delay_us = d_max / C_SOUND * 1e6
    if closure_max_us is None:
        closure_max_us = 0.15 * max_delay_us
    clo = closure_error_us(taus - tau_offsets, pairs=pairs)
    return dict(az_deg=float(np.rad2deg(np.arctan2(u[1], u[0]))),
                u_norm=float(np.linalg.norm(u)),
                resid_mm=resid_mm,
                closure_us=clo,
                closure_max_us=float(closure_max_us),
                max_delay_us=float(max_delay_us),
                consistent=bool(clo <= closure_max_us),
                peak_min=float(peaks.min()),
                taus=taus,
                pairs=pairs)


def srp_phat(block, pos=None, fs=FS_DEFAULT, grid_deg=None, f_lo=F_LO, f_hi=F_HI,
             nfft=None, normalize=False, tau_offsets=None):
    """Steered-response-power spatial spectrum over azimuth, PHAT-weighted.

    Independent of doa_least_squares: it scans candidate directions and sums
    the phase-aligned cross-spectra, rather than picking per-pair peaks. Its
    value is the whole curve -- a second lobe reveals an interferer that the
    least-squares fit would silently average into its single estimate.

    Returns (grid_deg, power). normalize=False (the default) returns RAW power,
    which is what you must accumulate across frames: min-max normalizing each
    frame first equalizes frames of very different SNR and biases the pooled
    peak. Normalize once, at display time, via normalize_power().

    Isolated cost, measured on the validated ReSpeaker capture with IDENTICAL
    frame selection on both arms (95 frames each): +88.0 deg raw versus
    +89.0 deg per-frame-normalized -- a 1.0 deg shift.

    An earlier version of this note claimed 8 deg (+96 vs +88). That was wrong:
    the +96 deg figure had been measured with per-frame normalization AND a
    mismatched frame selection (raw rms compared against a windowed-rms
    threshold), so it conflated two independent bugs. Most of the 8 deg was the
    selection mismatch, not the normalization.
    """
    block = np.asarray(block, dtype=np.float64)
    if pos is None:
        pos = array_positions()
    pos = np.asarray(pos, dtype=np.float64)
    if grid_deg is None:
        grid_deg = np.arange(0.0, 360.0, 1.0)
    pairs = PAIRS if pos.shape[0] == 4 else pairs_for(pos.shape[0])
    if tau_offsets is None:
        tau_offsets = np.zeros(len(pairs))
    tau_offsets = np.asarray(tau_offsets, dtype=np.float64)
    n = block.shape[1]
    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(n)))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    fb = freqs[band]
    X = np.fft.rfft(block * np.hanning(n)[None, :], nfft, axis=1)[:, band]
    dirs = np.stack([np.cos(np.deg2rad(grid_deg)), np.sin(np.deg2rad(grid_deg))], axis=1)
    # Steering delay per direction, plus the fixed instrumental delay of each
    # capture path: the measured cross-spectrum carries both.
    tau_grid = (dirs @ pair_baselines(pos, pairs).T) / C_SOUND + tau_offsets[None, :]
    power = np.zeros(len(grid_deg))
    for k, (i, j) in enumerate(pairs):
        R = X[i] * np.conj(X[j])
        R = R / np.maximum(np.abs(R), 1e-12)
        # MINUS in the exponent: R = X_i conj(X_j) carries phase
        # exp(+2j*pi*f*(p_i - p_j).u / c), so the steering vector must CONJUGATE
        # it to sum coherently at the true direction. The original used a plus
        # sign, which doubles the phase instead of cancelling it and peaks 180
        # deg away -- the same inversion gcc_phat had. Because both estimators
        # were wrong in the same direction they agreed with each other, so
        # cross-checking one against the other could never expose it.
        # tests/test_doa_core.py::test_srp_phat_peaks_at_truth covers this.
        power += np.real(np.exp(-2j * np.pi * fb[None, :] * tau_grid[:, k:k + 1])
                         * R[None, :]).sum(axis=1)
    return grid_deg, (normalize_power(power) if normalize else power)


class SRPAccumulator:
    """Pool raw SRP-PHAT power across frames and read the bearing off the peak.

    This is the ESTIMATOR, not a display smoother. It replaces "compute a
    least-squares bearing per frame, then circular-average the bearings", which
    is what the live display used to do and which is materially less accurate:
    a single reverberation-dominated frame produces a wild bearing that drags
    the circular mean, and no amount of averaging undoes it.

    Pooling power instead defers the argmax to the very end, so a bad frame
    contributes a low broad ridge rather than a confident wrong angle, and
    frames that agree reinforce each other. Measured on the four marked
    positions of the 2026-08-04 Panza session, with the sign and geometry
    faults already fixed:

        per-frame least-squares, circular mean   48.4 deg mean abs error
        pooled SRP (this class)                  13.0 deg
        pooled SRP + noise subtraction            7.5 deg

    Accumulate RAW power. Min-max normalizing each frame before summing
    equalizes frames of very different SNR and biases the pooled peak -- see
    the note in srp_phat(). Normalize once, at display time.

    Two pooling modes:
      decay=None   plain sum -- for offline scoring of a whole take.
      decay=tc_s   exponential forget with time constant tc_s -- for a live
                   display that must follow a moving talker.

    NOISE SUBTRACTION (subtract_noise=True) additionally accumulates a second
    spectrum over NON-speech frames and subtracts it, which removes a
    stationary interferer. On the 2026-08-04 session a fan or HVAC unit at
    ~280 deg dominated every take and this recovered 13.0 -> 7.5 deg. It needs
    the interferer to be genuinely stationary; it is a mitigation, not a
    substitute for silencing the source.
    """

    def __init__(self, pos, fs=FS_DEFAULT, grid_deg=None, f_lo=F_LO, f_hi=F_HI,
                 decay=None, hop=512, subtract_noise=False):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.fs = int(fs)
        self.grid = np.arange(0.0, 360.0, 1.0) if grid_deg is None else np.asarray(grid_deg, float)
        self.f_lo, self.f_hi = float(f_lo), float(f_hi)
        self.subtract_noise = bool(subtract_noise)
        self.speech = np.zeros(len(self.grid))
        self.noise = np.zeros(len(self.grid))
        self.n_speech = 0
        self.n_noise = 0
        self.alpha = None if decay is None else float(np.exp(-hop / (decay * self.fs)))

    def update(self, block, active=True):
        """Add one frame. `active` routes it to the speech or the noise pool."""
        _, p = srp_phat(block, pos=self.pos, fs=self.fs, grid_deg=self.grid,
                        f_lo=self.f_lo, f_hi=self.f_hi, normalize=False)
        if active:
            if self.alpha is None:
                self.speech += p
            else:
                self.speech = self.alpha * self.speech + (1 - self.alpha) * p
            self.n_speech += 1
        else:
            if self.alpha is None:
                self.noise += p
            else:
                self.noise = self.alpha * self.noise + (1 - self.alpha) * p
            self.n_noise += 1
        return p

    def spectrum(self):
        """Pooled spectrum, normalized to [0,1] for display/peak-picking."""
        if self.n_speech == 0:
            return np.zeros(len(self.grid))
        p = self.speech
        if self.subtract_noise and self.n_noise > 0:
            # Both pools are means when decaying, sums otherwise -- match scales.
            a = p / (1.0 if self.alpha is not None else max(self.n_speech, 1))
            b = self.noise / (1.0 if self.alpha is not None else max(self.n_noise, 1))
            p = a - b
        return normalize_power(p)

    def bearing(self):
        """(az_deg, sharpness) from the pooled peak.

        `sharpness` is 1 - (mean power / peak power) over the normalized
        spectrum: ~0 when the spectrum is flat (no usable direction), higher
        when one lobe dominates. It is NOT a confidence in degrees -- with a
        64 mm aperture the main lobe is ~130 deg wide no matter how good the
        frame is, so do not read a narrow beam into a high value.
        """
        p = self.spectrum()
        if p.max() <= 0:
            return float('nan'), 0.0
        return float(self.grid[int(np.argmax(p))]), float(1.0 - p.mean() / p.max())


def normalize_power(power):
    """Min-max a (possibly frame-accumulated) SRP curve to [0, 1] for display."""
    power = np.asarray(power, dtype=np.float64)
    span = power.max() - power.min()
    return (power - power.min()) / (span if span > 0 else 1.0)


def frame_rms(block, window=True):
    """Reference-mic rms for voice-activity gating.

    window=True applies the SAME Hann taper the estimators use. This matters:
    comparing an UNWINDOWED rms against a threshold computed from WINDOWED rms
    admits 153 frames where 95 belong, and moved the pooled SRP peak from the
    correct +88 deg to +92 deg. Use one convention for both the threshold and
    the test -- this helper exists so that is automatic.
    """
    x = np.asarray(block, dtype=np.float64)[0]
    if window:
        x = x * np.hanning(len(x))
    return float(np.sqrt((x ** 2).mean()))


def circular_mean(angles_deg, weights=None):
    """(mean_deg, R, sd_deg) for angles on a circle.

    A plain arithmetic mean is wrong here: it puts the mean of 359 and 1 at
    180. R in [0,1] is the concentration (1 = perfect agreement); sd_deg is
    the circular standard deviation.
    """
    a = np.asarray(angles_deg, dtype=np.float64)
    if a.size == 0:
        return float('nan'), 0.0, float('nan')
    z = np.exp(1j * np.deg2rad(a))
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        z = (z * w).sum() / max(w.sum(), 1e-12)
    else:
        z = z.mean()
    R = float(np.abs(z))
    # Clamp R to [eps, 1]: with tightly clustered angles floating-point can put
    # R a hair above 1, making log(R) positive and the sqrt argument negative
    # (RuntimeWarning, then NaN sd on the most confident frames of all).
    sd = float(np.rad2deg(np.sqrt(-2.0 * np.log(min(max(R, 1e-12), 1.0)))))
    return float(np.rad2deg(np.angle(z))), R, sd


class DriftMonitor:
    """MEASURE the delay drift between independent capture clocks. Does not fix it.

    Read this before using it
    -------------------------
    This class deliberately does not correct drift, because on TDOA data alone
    that is not possible, and an earlier version of this file that tried was
    wrong. The record of what was measured:

    A single multichannel device samples every mic off one crystal, so the only
    inter-channel delay is acoustic. N separate USB microphones are N
    independent crystals. 1 ppm of relative error is 1 us of drift per second,
    by definition, and cheap USB audio crystals run +-50 to +-100 ppm.

    Simulated on a 3-mic triangle (120/100 mm) with a STATIONARY source at
    90 deg and per-mic drifts of 0/+80/-40 ppm, over 12 s:

        one shared clock       : sd 0.22 deg, total swing  1.19 deg
        independent clocks     : sd 89.2 deg, total swing  359.19 deg

    Drift alone rotated the reported bearing by 195 deg with nothing moving.

    Why low-pass filtering the TDOA does NOT fix it: a stationary talker
    contributes a CONSTANT tau and linear drift contributes a slow RAMP, and
    both are "slow". Subtracting a smoothed tau removes the acoustic delay
    along with the drift. Measured: after such correction, true bearings of
    30/90/200 deg all read 283/309/318 deg -- the output no longer tracks the
    source at all.

    Why fitting a line to tau(t) does not fix it either: the GCC-PHAT search is
    bounded by the array aperture (correctly), so once accumulated drift
    approaches that bound the peak stops following the ramp and hops between
    reflections. Fitting the corrupted ramp recovered +22.67 ppm where the true
    relative drift was -80 ppm.

    The only correct fix is to remove drift in the TIME DOMAIN before
    correlation -- resample each stream onto a common clock using a rate
    estimated from a shared reference signal (a common loopback tone, or word
    clock). Failing that, use one multichannel interface.

    Usable window, at 5 deg of tolerated bearing error at broadside:

        baseline    1 ppm   10 ppm   50 ppm  100 ppm
         60 mm      15.2s     1.5s     0.3s     0.2s
        100 mm      25.4s     2.5s     0.5s     0.3s
        250 mm      63.5s     6.4s     1.3s     0.6s

    So report bearings over short windows, show the measured drift rate, and do
    not average over many seconds.
    """

    def __init__(self, n_pairs, fs=FS_DEFAULT, hop=512, max_step_us=60.0):
        self.n_pairs = int(n_pairs)
        self.fs = int(fs)
        self.hop = int(hop)
        self.max_step = max_step_us * 1e-6
        self.tau0 = None
        self.t0 = 0.0
        self.t = 0.0
        self.last_taus = None
        self.jump_detected = False
        self.n = 0

    def update(self, taus, active=True):
        """Record one frame's TDOAs. Returns the accumulated drift (seconds).

        The return value is diagnostic only -- do NOT pass it as tau_offsets to
        remove drift; see the class docstring for the measurement showing why
        that destroys the bearing.
        """
        taus = np.asarray(taus, dtype=np.float64)
        self.t += self.hop / self.fs
        if self.tau0 is None:
            self.tau0 = taus.copy()
            self.last_taus = taus.copy()
            self.t0 = self.t
            self.n = 1
            return np.zeros(self.n_pairs)
        if self.last_taus is not None:
            self.jump_detected = bool(
                np.any(np.abs(taus - self.last_taus) > self.max_step))
        self.last_taus = taus.copy()
        if active:
            self.n += 1
        return taus - self.tau0

    def drift_rate_ppm(self):
        """Apparent relative clock error per pair, ppm. NaN until 2 s of data.

        This is a LOWER BOUND on the true rate, and can badly understate it.
        Once accumulated drift approaches the array aperture, the bounded
        GCC-PHAT search stops following the ramp, so the endpoint difference
        saturates. Measured on the 0/+80/-40 ppm simulation, this reported
        -6.5 ppm for a pair whose true relative drift was -80 ppm.

        Use it as evidence that drift IS occurring and roughly how fast, never
        as a calibration constant. `saturating` tells you when to distrust it.
        """
        span = self.t - self.t0
        if self.tau0 is None or span < 2.0:
            return np.full(self.n_pairs, np.nan)
        return (self.last_taus - self.tau0) / span * 1e6

    def saturating(self, aperture_m):
        """True when accumulated drift is at/over the aperture the search allows.

        Past this point the TDOA no longer tracks the drift ramp and every
        derived number -- bearing AND drift rate -- is unreliable.
        """
        if self.tau0 is None or self.last_taus is None:
            return False
        return bool(np.any(np.abs(self.last_taus - self.tau0)
                           > 0.8 * aperture_m / C_SOUND))

    def usable_window_s(self, aperture_m, tol_deg=5.0):
        """Seconds before drift alone moves the bearing by tol_deg at broadside."""
        rate = np.nanmax(np.abs(self.drift_rate_ppm()))
        if not np.isfinite(rate) or rate <= 0:
            return np.inf
        dtau_us = (aperture_m / C_SOUND) * abs(
            np.cos(np.deg2rad(90.0 - tol_deg)) - np.cos(np.deg2rad(90.0))) * 1e6
        return dtau_us / rate


def calibrate_offset(block, pos, true_az_deg, fs=FS_DEFAULT, f_lo=F_LO, f_hi=F_HI):
    """Instrumental delay per pair, from a frame with a KNOWN source bearing.

    Put a talker (or a speaker playing noise) at a measured bearing, call this,
    and keep the result as tau_offsets. It converts a relative bearing-change
    display into an absolute one -- but only for as long as the clocks stay put,
    which for independent USB devices is seconds. Use it together with
    DriftTracker, not instead of it.
    """
    pos = np.asarray(pos, dtype=np.float64)
    pairs = PAIRS if pos.shape[0] == 4 else pairs_for(pos.shape[0])
    D = pair_baselines(pos, pairs)
    u = np.array([np.cos(np.deg2rad(true_az_deg)), np.sin(np.deg2rad(true_az_deg))])
    expected = (D @ u) / C_SOUND
    measured = np.array([gcc_phat(block[i], block[j], fs=fs, f_lo=f_lo, f_hi=f_hi)[0]
                         for i, j in pairs])
    return measured - expected


def aliasing_limit_hz(pos=None):
    """Highest unambiguous frequency, set by the LONGEST baseline."""
    if pos is None:
        pos = array_positions()
    pos = np.asarray(pos, dtype=np.float64)
    pairs = PAIRS if pos.shape[0] == 4 else pairs_for(pos.shape[0])
    d_max = max(np.linalg.norm(pos[i] - pos[j]) for i, j in pairs)
    return C_SOUND / (2.0 * d_max)


def beamwidth_deg(pos, f_hi=F_HI):
    """Approximate half-power beamwidth of the array, degrees.

    Rayleigh-style estimate: a source cannot be localized more finely than
    ~lambda/D at the highest usable frequency. Use it to scale any "is this a
    separate lobe?" test -- a fixed threshold in degrees is meaningless across
    arrays of different size.

    MEASURED on the ReSpeaker: lambda at 2600 Hz is 132 mm against a 64 mm
    aperture, so 0.5*lambda/d_max = 1.03 > 1 and the estimate SATURATES at its
    180 deg ceiling. That is not a rounding artefact -- it is the array
    reporting that it has no two-source resolving power at all, in the
    Rayleigh sense, at its own band limit. A single talker is still localised
    well (the pooled SRP peak is unbiased); what you cannot do is SEPARATE two
    talkers, which is exactly what the two-talker takes showed when 90 deg of
    separation merged into a single plateau.
    """
    pos = np.asarray(pos, dtype=np.float64)
    pairs = PAIRS if pos.shape[0] == 4 else pairs_for(pos.shape[0])
    d_max = max(np.linalg.norm(pos[i] - pos[j]) for i, j in pairs)
    lam = C_SOUND / float(f_hi)
    return float(np.rad2deg(2.0 * np.arcsin(np.clip(0.5 * lam / d_max, -1.0, 1.0))))


def secondary_peak(grid_deg, power, primary_deg, exclude_deg=25.0, pos=None,
                   f_hi=F_HI):
    """Strongest lobe at least `exclude_deg` away from the primary.

    Pass `pos` and the exclusion window is set to half the array's actual
    beamwidth instead of the fixed 25 deg default, which is the correct
    behaviour and what you almost always want.

    WHY THE DEFAULT IS A TRAP: 25 deg is far narrower than the main lobe of a
    small array, so the "second peak" it finds is the SHOULDER of the primary,
    not a second source. On the 2026-08-04 Panza session this reported
    "separation 26.0 deg" on both two-talker takes -- suspiciously just outside
    the 25 deg window, which is the signature of the artefact rather than a
    measurement. At 64 mm aperture and 2600 Hz beamwidth_deg() saturates at
    180 deg, so this array cannot resolve two talkers at all and ANY reported
    "second source" from it should be treated as a shoulder until proven
    otherwise.
    """
    if pos is not None:
        exclude_deg = max(exclude_deg, 0.5 * beamwidth_deg(pos, f_hi))
    grid_deg = np.asarray(grid_deg)
    off = np.abs(((grid_deg - primary_deg + 180.0) % 360.0) - 180.0)
    mask = off > exclude_deg
    if not mask.any():
        return float('nan'), 0.0
    idx = int(np.argmax(power[mask]))
    return float(grid_deg[mask][idx]), float(power[mask][idx])
