#!/usr/bin/env python3
"""
doa_session.py — run the dorai-phv.4 / TASK.md round-2 DoA protocol.

This automates everything around the human. It cannot walk to the floor marks
for you, and it does not pretend to: it prompts, records, scores, and writes
the bundle, and a person stands on the tape and talks.

    python3 doa_session.py --outdir ~/doa_sessions        # full protocol
    python3 doa_session.py --only t2_090                  # re-shoot one mark
    python3 doa_session.py --score ~/doa_sessions/session_2026-09-09
    python3 doa_session.py --list-devices

READ BEFORE RUNNING
-------------------
1.  STOP voice_mod FIRST. The ReSpeaker is a single UAC1.0 device and a single
    exclusive ALSA capture endpoint. This script opens it directly:

        docker compose stop voice doa wake

2.  Run the self-test first; the script does it for you and refuses to record
    if it is not 21/21. TASK.md Task 0: "If anything fails, stop and send us
    the output -- do not record." These are synthetic plane waves with known
    answers and they catch all three of the sign/geometry bugs instantly.

3.  Range 0.75 m, not 1.5 m. Mouth roughly level with the board. Mark EIGHT
    floor positions at 0/45/90/135/180/225/270/315 degrees from the board's
    MIC1 reference. The four diagonals are not optional: a reflected geometry
    is invisible at 0 and 180 because those lie ON the mirror axis, which is
    exactly how the last round's fit came out clean and wrong.

4.  Silence the fans before the noise_floor take.

SCORING
-------
The raw reported bearing, with NO rotation fitted, exactly as TASK.md
requires: "with the geometry now pinned by ground truth, a fitted offset would
hide exactly the class of error we are testing for. If a constant offset
reappears, that is a finding, not something to normalise away." The script
therefore reports the mean signed error alongside the mean absolute error, so
a constant offset is visible as a finding rather than silently removed.

Acceptance (Task 2): mean absolute error <= 15 deg, max <= 25 deg.
Task 3 additionally checks that t3_045 and t3_315 are not mirrored.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import wave
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import doa_core as dc                                        # noqa: E402

FS = 16000
BLOCK = 2048
HOP = 512                # offline: use the vendor's dense hop, CPU is free here
CHANNELS = (1, 2, 3, 4)  # the raw capsules; ch0 is the board DSP, ch5 loopback
CH_NAMES = {0: "aux_processed_device", 1: "mic1", 2: "mic2", 3: "mic3",
            4: "mic4", 5: "aux_playback"}

# tag -> (seconds, truth_deg or None, prompt)
PLAN = [
    ("noise_floor", 30, None,
     "EMPTY ROOM. Nobody speaks. Stand still and stay quiet."),
    ("t2_000", 10, 0.0, "Stand at the 0 deg mark, 0.75 m, talk continuously."),
    ("t2_045", 10, 45.0, "Stand at the 45 deg mark, talk continuously."),
    ("t2_090", 10, 90.0, "Stand at the 90 deg mark, talk continuously."),
    ("t2_135", 10, 135.0, "Stand at the 135 deg mark, talk continuously."),
    ("t2_180", 10, 180.0, "Stand at the 180 deg mark, talk continuously."),
    ("t2_225", 10, 225.0, "Stand at the 225 deg mark, talk continuously."),
    ("t2_270", 10, 270.0, "Stand at the 270 deg mark, talk continuously."),
    ("t2_315", 10, 315.0, "Stand at the 315 deg mark, talk continuously."),
    ("t4_walk", 20, None,
     "Start at 0 deg. Talk continuously. After ~5 s CLAP ONCE, walk briskly "
     "to 90 deg, CLAP ONCE on arrival, keep talking."),
    ("t5_090_270", 10, None, "TWO talkers, at 90 and 270 deg, both talking."),
    ("t5_090_180", 10, None, "TWO talkers, at 90 and 180 deg, both talking."),
    ("calib_a", 12, None,
     "Walk SLOWLY around the array while talking, as full a circle as you can. "
     "Then stand at the 0 deg mark and speak for 3 s."),
    ("calib_b", 12, None, "Repeat the walk, same as calib_a."),
    # Recorded LAST, so they are a held-out check on the geometry rather than
    # an input to it. This is the take that makes a mirror loud.
    ("t3_045", 10, 45.0, "MIRROR CHECK. Stand at 45 deg, talk continuously."),
    ("t3_315", 10, 315.0, "MIRROR CHECK. Stand at 315 deg, talk continuously."),
]
TRUTH = {t: d for t, _, d, _ in PLAN if d is not None}


# --------------------------------------------------------------------------
def _find(*rel):
    """Locate a packaged file in the source tree or the installed share dir."""
    cands = [os.path.join(os.path.dirname(_HERE), *rel)]
    try:
        from ament_index_python.packages import get_package_share_directory
        cands.append(os.path.join(get_package_share_directory("doa_mod"), *rel))
    except Exception:
        pass
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def self_test():
    """TASK.md Task 0. Returns (ok, output)."""
    test = _find("tests", "test_doa_core.py")
    if test is None:
        return False, "self-test test_doa_core.py not found"
    r = subprocess.run([sys.executable, test], capture_output=True, text=True,
                       cwd=os.path.dirname(test))
    out = (r.stdout or "") + (r.stderr or "")
    return ("21/21 passed" in out), out


def list_devices():
    import sounddevice as sd
    print(sd.query_devices())


def find_device(name_filter):
    import sounddevice as sd
    devs = sd.query_devices()
    need = max(CHANNELS) + 1
    cand = [(i, d) for i, d in enumerate(devs)
            if d.get("max_input_channels", 0) >= need
            and (not name_filter or name_filter.lower() in d["name"].lower())]
    if not cand:
        raise SystemExit(
            f"No input device with >= {need} channels matching "
            f"{name_filter!r}. Is voice_mod still holding the ReSpeaker? "
            f"`docker compose stop voice doa wake`. Inputs found:\n  "
            + "\n  ".join(f"#{i} '{d['name']}' ({d['max_input_channels']}ch)"
                          for i, d in enumerate(devs)
                          if d.get("max_input_channels", 0) > 0)
            + "\nMatch the PORTAUDIO name, not the ALSA card id: the "
              "ReSpeaker's card id is ArrayUAC10 but PortAudio calls it "
              "'ReSpeaker 4 Mic Array (UAC1.0)'. Card numbers move across "
              "reboots, so never pin one.")
    return cand[0]


def record(device, seconds, outdir, n_dev_ch):
    """Record `seconds` of the full device width; write one WAV per channel."""
    import sounddevice as sd
    os.makedirs(outdir, exist_ok=True)
    frames = []
    with sd.InputStream(device=device, channels=n_dev_ch, samplerate=FS,
                        dtype="float32", blocksize=1024) as st:
        t0 = time.time()
        while time.time() - t0 < seconds:
            blk, over = st.read(1024)
            if over:
                print("  ! input overflow", file=sys.stderr)
            frames.append(blk.copy())
    x = np.concatenate(frames, axis=0)          # [T, n_dev_ch]
    for c in range(x.shape[1]):
        name = CH_NAMES.get(c, f"ch{c}")
        path = os.path.join(outdir, f"ch{c}_{name}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(FS)
            w.writeframes((np.clip(x[:, c], -1, 1) * 32767).astype("<i2").tobytes())
    return x.shape[0] / float(FS)


def load_take(takedir):
    """[M, T] of the four capsules from a take directory."""
    chans = []
    for c in CHANNELS:
        path = os.path.join(takedir, f"ch{c}_{CH_NAMES[c]}.wav")
        if not os.path.exists(path):
            return None
        with wave.open(path, "rb") as w:
            n = w.getnframes()
            raw = np.frombuffer(w.readframes(n), dtype="<i2")
        chans.append(raw.astype(np.float64) / 32768.0)
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans], axis=0)


def score_take(block_mc, noise_pool=None, radius_mm=32.0, f_hi=dc.F_HI):
    """Pooled SRP-PHAT bearing over a whole take. No rotation fitted.

    decay=None: a plain sum, which is the right pooling for offline scoring of
    a complete take (the live node uses an exponential forget so it can follow
    a talker who moves). Raw power in, normalise once at the end -- per-frame
    normalisation equalises frames of very different SNR and moves the peak.
    """
    pos = dc.array_positions(radius_mm / 1000.0)
    acc = dc.SRPAccumulator(pos, fs=FS, f_hi=f_hi, decay=None,
                            subtract_noise=noise_pool is not None)
    if noise_pool is not None:
        acc.noise = noise_pool[0].copy()
        acc.n_noise = noise_pool[1]
    try:
        from vad import NeuralVAD
        vad = NeuralVAD(_find("vad.onnx"))
    except Exception as exc:
        print(f"  ! neural VAD unavailable ({exc}); using an RMS energy gate")
        vad = None
    rms_hist = []
    n_act = n_ok = 0
    for s in range(0, block_mc.shape[1] - BLOCK + 1, HOP):
        blk = block_mc[:, s:s + BLOCK]
        if vad is not None:
            active = vad.process_buffer(blk[0], fs=FS)["is_speech"]
        else:
            r = dc.frame_rms(blk)
            rms_hist.append(r)
            active = len(rms_hist) >= 20 and r > np.percentile(rms_hist, 70.0)
        est = dc.doa_least_squares(blk, pos=pos, f_hi=f_hi, fs=FS)
        if active:
            n_act += 1
            n_ok += 1 if est.get("consistent", True) else 0
        acc.update(blk, active=active)
    az, sharp = acc.bearing()
    return dict(bearing_deg=az, sharpness=sharp, n_speech=acc.n_speech,
                n_noise=acc.n_noise, n_active=n_act, n_closure_ok=n_ok,
                spectrum=acc.spectrum().tolist())


def noise_pool_from(takedir):
    mc = load_take(takedir)
    if mc is None:
        return None
    pos = dc.array_positions()
    acc = dc.SRPAccumulator(pos, fs=FS, decay=None)
    for s in range(0, mc.shape[1] - BLOCK + 1, HOP):
        acc.update(mc[:, s:s + BLOCK], active=False)
    return (acc.noise, acc.n_noise)


def ang_err(reported, truth):
    return float((reported - truth + 180.0) % 360.0 - 180.0)


def score_session(root, subtract_noise=True):
    nf = os.path.join(root, "noise_floor")
    pool = noise_pool_from(nf) if (subtract_noise and os.path.isdir(nf)) else None
    if subtract_noise and pool is None:
        print("! no noise_floor take; scoring WITHOUT interferer subtraction")
    results = {}
    for tag in sorted(os.listdir(root)):
        d = os.path.join(root, tag)
        if not os.path.isdir(d) or tag == "noise_floor":
            continue
        mc = load_take(d)
        if mc is None:
            continue
        r = score_take(mc, noise_pool=pool)
        if tag in TRUTH:
            r["truth_deg"] = TRUTH[tag]
            r["error_deg"] = ang_err(r["bearing_deg"], TRUTH[tag])
        results[tag] = r
        e = f"{r['error_deg']:+7.1f}" if "error_deg" in r else "      -"
        print(f"  {tag:12s} bearing {r['bearing_deg']:6.1f} deg  err {e}  "
              f"sharpness {r['sharpness']:.3f}  "
              f"closure {r['n_closure_ok']}/{max(r['n_active'], 1)}")

    t2 = [r for t, r in results.items()
          if t.startswith("t2_") and "error_deg" in r]
    summary = {}
    if t2:
        errs = np.array([abs(r["error_deg"]) for r in t2])
        signed = np.array([r["error_deg"] for r in t2])
        mean_signed, R, _ = dc.circular_mean(signed)
        summary = dict(n=len(t2), mean_abs_err=float(errs.mean()),
                       max_abs_err=float(errs.max()),
                       mean_signed_err=float(mean_signed),
                       pass_task2=bool(errs.mean() <= 15.0 and errs.max() <= 25.0))
        print(f"\nTask 2 over {len(t2)} marks: mean |err| "
              f"{errs.mean():.1f} deg (<= 15 required), max {errs.max():.1f} deg "
              f"(<= 25 required) -> {'PASS' if summary['pass_task2'] else 'FAIL'}")
        print(f"Mean SIGNED error {mean_signed:+.1f} deg (R={R:.2f}). A large "
              f"consistent value here is a FINDING -- a constant frame offset "
              f"-- and is deliberately not fitted out.")
    if "t3_045" in results and "t3_315" in results:
        a = results["t3_045"]["bearing_deg"]
        b = results["t3_315"]["bearing_deg"]
        mirrored = (abs(ang_err(a, 315.0)) < 25.0 and abs(ang_err(b, 45.0)) < 25.0)
        ok = (abs(ang_err(a, 45.0)) <= 25.0 and abs(ang_err(b, 315.0)) <= 25.0)
        summary["mirror_check"] = "MIRRORED" if mirrored else ("PASS" if ok else "FAIL")
        print(f"Task 3 mirror check: t3_045 -> {a:.0f}, t3_315 -> {b:.0f}  "
              f"=> {summary['mirror_check']}")
        if mirrored:
            print("  The geometry is still reflected. Everything else in this "
                  "session is suspect.")
    with open(os.path.join(root, "scores.json"), "w") as f:
        json.dump({"takes": results, "summary": summary}, f, indent=2)
    print(f"\nWrote {os.path.join(root, 'scores.json')}")
    return summary


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.path.expanduser("~/doa_sessions"))
    # A PortAudio name substring, not the ALSA card id: the card id is
    # ArrayUAC10 but PortAudio calls it "ReSpeaker 4 Mic Array (UAC1.0)".
    ap.add_argument("--device", default="ReSpeaker",
                    help="PortAudio device-name substring (default ReSpeaker)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="record only these tags, into an existing session")
    ap.add_argument("--session", default=None,
                    help="reuse this session directory instead of a new one")
    ap.add_argument("--score", default=None, metavar="SESSION_DIR",
                    help="score an existing session and exit; records nothing")
    ap.add_argument("--no-subtract-noise", action="store_true")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--skip-self-test", action="store_true",
                    help="TASK.md says do not. Here only for scoring reruns.")
    a = ap.parse_args(argv)

    if a.list_devices:
        list_devices()
        return 0

    if a.score:
        return 0 if score_session(a.score, not a.no_subtract_noise) else 0

    print("=" * 72)
    print("Task 0 — self-test (synthetic plane waves, known answers)")
    ok, out = self_test()
    print(out.strip().splitlines()[-1] if out.strip() else "(no output)")
    if not ok and not a.skip_self_test:
        print(out)
        print("\nSTOP. TASK.md: if the self-test does not pass 21/21, do not "
              "record. Send this output upstream.")
        return 1

    dev_idx, dev = find_device(a.device)
    n_dev_ch = int(dev["max_input_channels"])
    print(f"\nDevice #{dev_idx} '{dev['name']}' ({n_dev_ch}ch @ {FS} Hz)")

    root = a.session or os.path.join(
        a.outdir, "session_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(root, exist_ok=True)
    print(f"Session: {root}")

    plan = [p for p in PLAN if (a.only is None or p[0] in a.only)]
    if not plan:
        print(f"No plan entries match --only {a.only}")
        return 1

    print("\nSetup, from TASK.md: range 0.75 m, mouth level with the board, "
          "eight marks at 0/45/90/135/180/225/270/315 deg from MIC1, fans OFF, "
          "and photograph the array with the 0 deg mark visible.\n")

    for tag, secs, truth, prompt in plan:
        d = os.path.join(root, tag)
        print("-" * 72)
        t = f"  (ground truth {truth:.0f} deg)" if truth is not None else ""
        print(f"{tag}{t} — {secs} s")
        print(f"  {prompt}")
        try:
            input("  Press Enter when in position (Ctrl+C to stop the session) ")
        except (EOFError, KeyboardInterrupt):
            print("\nSession stopped by operator.")
            break
        for k in (3, 2, 1):
            print(f"  {k}...", flush=True)
            time.sleep(1.0)
        print("  RECORDING", flush=True)
        try:
            got = record(dev_idx, secs, d, n_dev_ch)
        except KeyboardInterrupt:
            print("\n  aborted")
            continue
        print(f"  done, {got:.1f} s -> {d}")

    notes = os.path.join(root, "notes.txt")
    if not os.path.exists(notes):
        with open(notes, "w") as f:
            f.write("Task 0 self-test output\n"
                    "-----------------------\n" + out + "\n\n"
                    "Fill in before sending the bundle:\n"
                    "1. Was the ~280 deg interferer found, identified, silenced?\n"
                    "2. Talker height, and any deviation from 0.75 m.\n"
                    "3. Anything that felt wrong during the session.\n")
        print(f"\nWrote {notes} — fill in sections 1-3 by hand.")

    print("\n" + "=" * 72)
    print("Scoring — raw reported bearing, NO rotation fitted")
    score_session(root, not a.no_subtract_noise)
    return 0


if __name__ == "__main__":
    sys.exit(main())
