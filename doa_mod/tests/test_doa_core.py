"""Synthetic ground-truth regression tests for doa_core.

WHY THIS FILE EXISTS
--------------------
Two sign faults shipped undetected for months because every validation ran on
FIELD RECORDINGS, where an algorithm bug, a wiring convention, room
reverberation and a mirrored geometry are all confounded -- and no amount of
field data separates them.

Both bugs are caught in under a second by a synthetic plane wave with a known
answer, which is what this file is:

  1. gcc_phat returned the NEGATIVE of its documented tau convention.
  2. srp_phat carried the SAME inversion in its steering exponent. Because both
     estimators were wrong in the same direction they AGREED WITH EACH OTHER,
     so the original "the two methods agree to 1.0 deg" validation could never
     have exposed it. Agreement between two implementations of the same
     convention is not evidence of correctness.
  3. array_positions() was mirrored relative to the Seeed board.

Run:  python3 -m pytest tests/test_doa_core.py -v
      python3 tests/test_doa_core.py          # no pytest needed
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import doa_core as dc                                              # noqa: E402

FS = 16000
N = 2048


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def band_noise(n_samples, seed=0, f_lo=dc.F_LO, f_hi=dc.F_HI, fs=FS):
    """Broadband noise limited to the analysis band -- a stand-in for speech."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples * 2)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    X[(f < f_lo) | (f > f_hi)] = 0.0
    return np.fft.irfft(X)[:n_samples]


def delay(x, d_samples):
    """Fractional delay via FFT phase ramp. Positive d => x arrives LATER."""
    n = len(x)
    Nf = 4 * n
    X = np.fft.rfft(x, Nf)
    f = np.fft.rfftfreq(Nf, 1.0)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * d_samples), Nf)[:n]


def plane_wave(az_deg, pos, src=None, fs=FS, n=N):
    """[M,n] block for a far-field source at `az_deg`, exact geometry.

    Arrival time at mic k is t_k = -(p_k . u)/c: the mic CLOSEST to the source
    (largest p_k . u) arrives FIRST. Getting this backwards is precisely the
    bug the suite exists to catch, so it is written out explicitly here.
    """
    if src is None:
        src = band_noise(n * 2, fs=fs)
    u = np.array([np.cos(np.deg2rad(az_deg)), np.sin(np.deg2rad(az_deg))])
    t = -(np.asarray(pos) @ u) / dc.C_SOUND
    t = t - t.min() + 1e-3                     # keep all delays positive
    return np.stack([delay(src, ti * fs)[:n] for ti in t])


def wrap180(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# 1. the sign convention -- the test that would have caught bug #1
# ---------------------------------------------------------------------------
def test_gcc_phat_sign():
    """tau > 0 must mean b LAGS a, exactly as the docstring says."""
    a = band_noise(N)
    for d_us in (-150.0, -100.0, -25.0, 0.0, 25.0, 100.0, 150.0):
        b = delay(a, d_us * 1e-6 * FS)
        tau, _ = dc.gcc_phat(a, b, fs=FS)
        assert abs(tau * 1e6 - d_us) < 2.0, (
            f"b lags a by {d_us:+.0f} us but gcc_phat returned "
            f"{tau*1e6:+.1f} us. If this is the exact NEGATIVE, the lag-axis "
            f"index mapping in gcc_phat has been reverted -- see the SIGN "
            f"CONVENTION note there.")


def test_gcc_phat_identical_signals_give_zero():
    a = band_noise(N)
    tau, _ = dc.gcc_phat(a, a.copy(), fs=FS)
    assert abs(tau) < 2e-6


# ---------------------------------------------------------------------------
# 2. geometry -- the test that would have caught bug #3
# ---------------------------------------------------------------------------
def test_array_positions_matches_board_drawing():
    """ch1..ch4 at 45/135/225/315 deg CCW, per the Seeed drawing.

    The previous value [135, 45, -45, -135] is this layout MIRRORED about the
    x-axis. A mirror is NOT a rotation and cannot be calibrated away with a
    constant offset -- it must be right in the geometry.
    """
    pos = dc.array_positions()
    ang = np.degrees(np.arctan2(pos[:, 1], pos[:, 0])) % 360.0
    np.testing.assert_allclose(ang, [45.0, 135.0, 225.0, 315.0], atol=1e-6)
    # identical to the generic circle helper
    np.testing.assert_allclose(
        pos, dc.circle_positions(4, dc.R_NOMINAL, start_deg=45.0), atol=1e-12)


def test_array_positions_is_a_square_of_the_right_size():
    pos = dc.array_positions()
    d = sorted(np.linalg.norm(pos[i] - pos[j]) * 1000 for i, j in dc.PAIRS)
    np.testing.assert_allclose(d[:4], [45.25] * 4, atol=0.1)   # sides
    np.testing.assert_allclose(d[4:], [64.0] * 2, atol=0.1)    # diagonals


# ---------------------------------------------------------------------------
# 3. end-to-end: both estimators must recover a KNOWN azimuth
# ---------------------------------------------------------------------------
def test_least_squares_recovers_known_azimuth():
    pos = dc.array_positions()
    src = band_noise(N * 2)
    for az in np.arange(0.0, 360.0, 15.0):
        est = dc.doa_least_squares(plane_wave(az, pos, src), pos=pos, fs=FS)
        err = wrap180(est['az_deg'] - az)
        assert abs(err) < 1.5, f"true {az} deg -> {est['az_deg']%360:.1f} deg"


def test_srp_phat_peaks_at_truth():
    """SRP must peak at the TRUE azimuth, not 180 deg away.

    Guards the steering-exponent sign independently of gcc_phat. These two
    were wrong in the SAME direction and therefore agreed with each other --
    which is why cross-validating one against the other proved nothing.
    """
    pos = dc.array_positions()
    src = band_noise(N * 2)
    for az in np.arange(0.0, 360.0, 30.0):
        grid, p = dc.srp_phat(plane_wave(az, pos, src), pos=pos, fs=FS)
        assert abs(wrap180(grid[int(np.argmax(p))] - az)) < 3.0


def test_both_estimators_agree_AND_are_correct():
    """Agreement alone is not the test -- both must also match ground truth."""
    pos = dc.array_positions()
    src = band_noise(N * 2)
    for az in (37.0, 122.0, 251.0, 318.0):
        block = plane_wave(az, pos, src)
        ls = dc.doa_least_squares(block, pos=pos, fs=FS)['az_deg']
        grid, p = dc.srp_phat(block, pos=pos, fs=FS)
        assert abs(wrap180(ls - az)) < 1.5
        assert abs(wrap180(grid[int(np.argmax(p))] - az)) < 3.0


def test_plane_wave_solution_is_unit_norm():
    """|u| = cos(elevation) = 1 for a source in the array plane."""
    pos = dc.array_positions()
    est = dc.doa_least_squares(plane_wave(75.0, pos), pos=pos, fs=FS)
    assert abs(est['u_norm'] - 1.0) < 0.02
    assert est['resid_mm'] < 0.5


def test_estimators_are_invariant_to_radius_scaling():
    """Azimuth is a ratio of direction components, so it must not move."""
    src = band_noise(N * 2)
    for r in (0.028, 0.032, 0.036):
        pos = dc.array_positions(r)
        est = dc.doa_least_squares(plane_wave(210.0, pos, src), pos=pos, fs=FS)
        assert abs(wrap180(est['az_deg'] - 210.0)) < 1.5


# ---------------------------------------------------------------------------
# 4. mirror detection -- the failure mode the field harness could not see
# ---------------------------------------------------------------------------
def test_mirrored_geometry_produces_mirrored_bearings():
    """A reflected `pos` maps every bearing to -theta.

    This is why Panza's rotation-fit could not distinguish the two candidate
    geometries: a fit that searches over reflection absorbs it into its own
    mirror branch and reports an IDENTICAL residual. Only ground truth -- an
    absolute reference like this test, or marked positions off the mirror
    axis -- can break the tie.
    """
    pos = dc.array_positions()
    pos_mirror = pos * np.array([1.0, -1.0])
    src = band_noise(N * 2)
    for az in (30.0, 60.0, 120.0, 300.0):
        block = plane_wave(az, pos, src)
        good = dc.doa_least_squares(block, pos=pos, fs=FS)['az_deg']
        bad = dc.doa_least_squares(block, pos=pos_mirror, fs=FS)['az_deg']
        assert abs(wrap180(good - az)) < 1.5
        assert abs(wrap180(bad - (-az))) < 1.5


def test_marks_on_the_mirror_axis_cannot_detect_a_mirror():
    """0 and 180 deg are FIXED POINTS of the reflection -- they fit either way.

    Encoded as a test so nobody re-runs an acceptance protocol using only
    marks on the axis. Any geometry check must include off-axis positions;
    the protocol therefore calls for 45/135/225/315 deg as well.
    """
    pos = dc.array_positions()
    pos_mirror = pos * np.array([1.0, -1.0])
    src = band_noise(N * 2)
    for az in (0.0, 180.0):
        block = plane_wave(az, pos, src)
        a = dc.doa_least_squares(block, pos=pos, fs=FS)['az_deg']
        b = dc.doa_least_squares(block, pos=pos_mirror, fs=FS)['az_deg']
        assert abs(wrap180(a - b)) < 2.0, "on-axis marks should be ambiguous"


# ---------------------------------------------------------------------------
# 5. closure -- the geometry-free frame-quality gate
# ---------------------------------------------------------------------------
def test_closure_is_zero_for_a_clean_plane_wave():
    pos = dc.array_positions()
    est = dc.doa_least_squares(plane_wave(95.0, pos), pos=pos, fs=FS)
    assert est['closure_us'] < 1.0
    assert est['consistent']


def test_closure_is_blind_to_a_wrong_geometry():
    """Closure responds to the FRAME, never to `pos` -- by construction."""
    pos = dc.array_positions()
    block = plane_wave(95.0, pos)
    a = dc.doa_least_squares(block, pos=pos, fs=FS)
    for wrong in (pos * np.array([1.0, -1.0]),          # mirrored
                  pos * np.array([1.0, 0.45]),          # squashed
                  pos * 1.6,                            # rescaled
                  pos[[1, 0, 2, 3]]):                   # re-labelled
        b = dc.doa_least_squares(block, pos=wrong, fs=FS)
        assert abs(a['closure_us'] - b['closure_us']) < 1e-6


def test_residual_cannot_detect_ANY_linear_geometry_error():
    """The plane-wave residual is blind to every invertible linear map of `pos`.

    THIS IS THE RESULT THAT EXPLAINS THE 2026-08-04 MISDIAGNOSIS, and it is
    much stronger than "blind to a mirror".

    Proof: the residual is the distance from the measured delay vector d to the
    column space of D, whose rows are the baselines p_i - p_j. Replacing
    pos -> pos @ A.T for an invertible 2x2 A gives D' = D @ A.T, whose columns
    are linear combinations of D's columns -- so col(D') == col(D) EXACTLY, and
    the distance from d to it is unchanged.

    Consequently the residual cannot see a reflection, a rotation, a rescaling,
    a shear, or any composition of them. It reacts ONLY to channel
    re-labellings that are not realisable as a linear map. A field report that
    uses residual (or a rotation-fit built on it) to choose between two
    candidate geometries is therefore reading a number that is constant across
    the entire hypothesis class it is trying to discriminate.

    Breaking the tie needs an ABSOLUTE reference: a synthetic source with a
    known answer (this suite), or marked talker positions OFF the mirror axis.
    """
    pos = dc.array_positions()
    block = plane_wave(95.0, pos)
    base = dc.doa_least_squares(block, pos=pos, fs=FS)['resid_mm']
    for A in (np.array([[1.0, 0.0], [0.0, -1.0]]),      # reflection
              np.array([[0.0, -1.0], [1.0, 0.0]]),      # 90 deg rotation
              np.array([[1.6, 0.0], [0.0, 1.6]]),       # scaling
              np.array([[1.0, 0.6], [0.0, 1.0]]),       # shear
              np.array([[0.3, 1.7], [-0.9, 0.4]])):     # arbitrary
        r = dc.doa_least_squares(block, pos=pos @ A.T, fs=FS)['resid_mm']
        assert abs(r - base) < 1e-9, f"residual moved under {A.tolist()}"


def test_a_relabelling_that_mirrors_is_invisible_to_the_residual():
    """Reversing channel order fits PERFECTLY yet reports the mirrored bearing.

    This is the concrete trap: ch1..ch4 wired backwards is indistinguishable
    from correct wiring by any goodness-of-fit measure, and reports 265 deg for
    a source at 95 deg.
    """
    pos = dc.array_positions()
    block = plane_wave(95.0, pos)
    good = dc.doa_least_squares(block, pos=pos, fs=FS)
    rev = dc.doa_least_squares(block, pos=pos[[3, 2, 1, 0]], fs=FS)
    assert abs(rev['resid_mm'] - good['resid_mm']) < 1e-6
    assert abs(wrap180(rev['az_deg'] - (-95.0))) < 2.0


def test_closure_rejects_incoherent_audio():
    """Independent noise per mic has no single source and must be rejected."""
    rng = np.random.default_rng(7)
    pos = dc.array_positions()
    block = np.stack([band_noise(N, seed=int(k)) for k in rng.integers(0, 1e6, 4)])
    est = dc.doa_least_squares(block, pos=pos, fs=FS)
    assert not est['consistent']


# ---------------------------------------------------------------------------
# 6. pooled SRP beats per-frame bearing averaging
# ---------------------------------------------------------------------------
def test_srp_accumulator_recovers_known_azimuth():
    pos = dc.array_positions()
    acc = dc.SRPAccumulator(pos, fs=FS)
    for seed in range(8):
        acc.update(plane_wave(140.0, pos, band_noise(N * 2, seed=seed)))
    az, sharp = acc.bearing()
    assert abs(wrap180(az - 140.0)) < 3.0
    assert sharp > 0.1


def test_srp_accumulator_survives_a_corrupted_frame():
    """One incoherent frame must not capture the pooled estimate.

    This is the concrete reason the live display moved off circular-averaging
    per-frame bearings: a single bad frame contributes a wild ANGLE to a mean,
    but only a low broad ridge to pooled POWER.
    """
    pos = dc.array_positions()
    rng = np.random.default_rng(3)
    acc = dc.SRPAccumulator(pos, fs=FS)
    per_frame = []
    for seed in range(8):
        block = (plane_wave(140.0, pos, band_noise(N * 2, seed=seed)) if seed != 4
                 else np.stack([band_noise(N, seed=900 + int(k))
                                for k in rng.integers(0, 1e6, 4)]))
        acc.update(block)
        per_frame.append(dc.doa_least_squares(block, pos=pos, fs=FS)['az_deg'])
    pooled, _ = acc.bearing()
    assert abs(wrap180(pooled - 140.0)) < 5.0
    mean_az, _, _ = dc.circular_mean(per_frame)
    assert abs(wrap180(pooled - 140.0)) <= abs(wrap180(mean_az - 140.0)) + 1e-9


def test_accumulator_subtracts_a_stationary_interferer():
    """Talker at 140 deg, continuous interferer at 280 deg (the fan case)."""
    pos = dc.array_positions()
    acc = dc.SRPAccumulator(pos, fs=FS, subtract_noise=True)
    for seed in range(10):
        noise = plane_wave(280.0, pos, band_noise(N * 2, seed=100 + seed))
        acc.update(noise, active=False)
        acc.update(noise + 1.2 * plane_wave(140.0, pos,
                                            band_noise(N * 2, seed=seed)),
                   active=True)
    az, _ = acc.bearing()
    assert abs(wrap180(az - 140.0)) < 20.0


# ---------------------------------------------------------------------------
# 7. beamwidth-aware peak picking
# ---------------------------------------------------------------------------
def test_beamwidth_saturates_for_this_array():
    """At 64 mm and 2600 Hz, lambda (132 mm) EXCEEDS twice the aperture.

    The Rayleigh estimate saturates at its 180 deg ceiling, which is the
    array honestly reporting that it has no two-source resolving power at its
    own band limit. A single talker is still localised fine; two talkers 90
    deg apart merge into one plateau, exactly as observed in the field.

    The old fixed exclude_deg=25 was therefore ~7x narrower than the main
    lobe, which is why "second peak" found the primary's own shoulder.
    """
    pos = dc.array_positions()
    assert dc.beamwidth_deg(pos) == 180.0
    # a much larger array does resolve: 300 mm aperture at 2600 Hz
    wide = dc.circle_positions(4, 0.15, start_deg=45.0)
    assert 20.0 < dc.beamwidth_deg(wide) < 90.0


def test_secondary_peak_with_pos_widens_the_exclusion():
    pos = dc.array_positions()
    grid = np.arange(0.0, 360.0, 1.0)
    power = np.exp(-0.5 * (wrap180(grid - 90.0) / 55.0) ** 2)   # ONE broad lobe
    _, p_default = dc.secondary_peak(grid, power, 90.0)
    _, p_aware = dc.secondary_peak(grid, power, 90.0, pos=pos)
    assert p_aware < p_default, "aperture-aware exclusion must reject the shoulder"


# ---------------------------------------------------------------------------
def _main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            bad += 1
            print(f"  FAIL  {name}\n        {exc}")
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(_main())
