#!/usr/bin/env python3
"""
wake_mod / wake_node.py — wake-word gate on /dorai/wake.

    /dorai_clean_audio  (beamformed mono, 16 kHz)
        -> Vosk KaldiRecognizer constrained to a ONE-WORD grammar
        -> /dorai/wake   std_msgs/String carrying the DEMO_INTERFACES JSON,
                         with the bearing already paired on

NO NEW DEPENDENCY. Vosk is already in the image and stt_mod already builds a
KaldiRecognizer at 16 kHz against the same model; this node builds a second
one over the same `Model` object with a grammar attached. A grammar-constrained
recogniser is a different machine from a dictation recogniser: the decoder can
only emit the keyword or `[unk]`, so instead of asking "did the transcript
happen to contain this word" -- which fires on anything the language model
finds plausible -- the acoustic score has to beat the explicit garbage model.
That is what makes a five-minute conversation quiet.

THE DECISION RULE, AND WHY IT IS NOT "DID A PARTIAL CONTAIN THE WORD"
---------------------------------------------------------------------
A wake event needs an ENDPOINTED result whose entire utterance is the keyword
and nothing else, at word confidence >= `min_conf`. Not a partial hypothesis,
and not "the keyword appeared somewhere in a longer result". This was measured,
not assumed. Against 4.1 minutes of continuous speech that never says the
keyword, and 5 spoken keywords:

    rule                                   keyword found   false alarms
    partial contains the keyword               5 / 5           28
    endpointed result contains it              5 / 5            3
    ... with word confidence >= 0.99           5 / 5            1
    endpointed result IS the keyword alone     5 / 5            0

The partial rule is the obvious one and it is unusable: mid-utterance the
decoder briefly hypothesises the keyword inside all sorts of words, then
retracts it, and by then the event has been published. The isolated-utterance
rule survives because in ordinary conversation the keyword never occupies a
whole endpointed segment on its own -- which is exactly how a person says a
wake word: alone, then a pause. That also makes the rule keyword-independent;
every candidate below scored 0 false alarms under it, so the keyword could be
chosen on acoustics rather than on this arithmetic.

The cost is latency: Vosk endpoints about 0.8 s after the word ends (measured
0.73-0.87 s). The published `stamp` is unaffected -- it is the word's own
`end` time, converted to epoch, not the time we noticed.

THE KEYWORD
-----------
Default `marvin`, and the choice is constrained harder than it looks:

  * It must be IN THE MODEL'S LEXICON. `artaban` is not, and Vosk says so out
    loud -- "Ignoring word missing in vocabulary: 'artaban'" -- and then
    silently builds a grammar containing only `[unk]`, which never fires.
    Verified against vosk-model-small-en-us-0.15 on atb. Using the robot's
    own name needs a custom lexicon entry (g2p the pronunciation into the
    model's dictionary and rebuild Gr.fst); that is a real option, and a
    separate piece of work.
  * Two syllables, as specified. One syllable is too easy for a noise
    transient to fake; three drifts into phrase territory.
  * It has to survive a noisy room, which is a phonetic requirement, not a
    semantic one. `marvin` is sonorant throughout -- /m/ nasal onset, the
    /ar/ rhotic nucleus, then /vin/ -- so almost all of its energy is voiced
    and sits in the 200-1000 Hz band where a beamformed 16 kHz stream is
    strongest and where broadband machine noise is weakest. That is the same
    reason it appears in the standard Speech Commands wake-word set. It also
    scored ZERO raw hits over the 4.1 minute negative take even under the
    loosest rule above, the widest margin of anything tested.
  * Rejected alternatives: `robot` is in vocabulary but is the single most
    frequent word in a robotics demo, so it is a false-trigger machine;
    `falcon` (9 raw false hits) opens on /f/, a weak fricative broadband
    noise eats; `condor` (3) and `orbit` (2) likewise scored worse raw;
    `compass` scored 0 but is a word people actually say about navigation;
    `jarvis` scored 0 and would work, but is somebody else's assistant.

BEARING PAIRING LIVES HERE
--------------------------
DEMO_INTERFACES puts the pairing in this node on purpose, so the bridge stays
a dumb pipe and the behaviour node stays dumb. We keep a ring of recent
/dorai/doa messages and attach the one whose `stamp` is nearest the wake
word's own audio window -- not the newest one, and not the one that happens to
arrive next. Those differ: the clean-audio path is a whole-signal beamformer,
so a wake word is recognised something like a second after it was spoken,
while the bearing for that same instant was published almost in real time and
is already several messages back in the ring. Matching on stamps is what makes
the robot turn toward where the talker was when they spoke.

Both stamps are epoch seconds describing the ACOUSTIC instant, because both
topics derive theirs from the same `capture_s` ADC timeline that voice_mod
puts in its 4-float header, so they are directly comparable.
"""

import json
import os
import sys
import time
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
except Exception:                                            # pragma: no cover
    Model = None


SCHEMA_V = 1


class WakeMod(Node):
    def __init__(self):
        super().__init__("wake_mod")
        from rcl_interfaces.msg import ParameterDescriptor
        dyn = ParameterDescriptor(dynamic_typing=True)

        self.declare_parameter("input_topic", "/dorai_clean_audio")
        self.declare_parameter("doa_topic", "/dorai/doa")
        self.declare_parameter("output_topic", "/dorai/wake")
        # See THE KEYWORD above before changing this: it must be in the Vosk
        # model's lexicon or the grammar silently degenerates to [unk] only.
        # The node checks and refuses to start rather than pretending to work.
        self.declare_parameter("keyword", "marvin")
        # Word confidence floor on the endpointed result. Every true detection
        # measured came back at exactly 1.0; the false alarms that survived
        # the isolation rule in earlier experiments sat at 0.55-0.9.
        self.declare_parameter("min_conf", 0.9, dyn)
        # Require the endpointed utterance to be the keyword ALONE. This is
        # the rule that took false alarms to zero; see the module docstring
        # for the measurements. Setting it false accepts the keyword anywhere
        # inside a result, which is much more permissive and was measured to
        # false-trigger. Only turn it off if you need "marvin, come here" to
        # fire, and expect to pay for it.
        self.declare_parameter("require_isolated", True)
        self.declare_parameter("model_lang", "en-us")
        self.declare_parameter("model_path", "")
        # Refractory period after a trigger. A wake word said once produces
        # one event; without this the same utterance re-fires as the partial
        # is re-emitted on the next chunk.
        self.declare_parameter("refractory_s", 2.0, dyn)
        # How far from the wake word's audio window a /dorai/doa message may
        # sit and still be its bearing. 1.5 s is a little over one avg_tc of
        # the pooled SRP accumulator.
        self.declare_parameter("pair_window_s", 1.5, dyn)
        # Sub-chunk fed to the recogniser between partial checks. This is the
        # resolution of the reported stamp, and 0.1 s costs nothing.
        self.declare_parameter("chunk_s", 0.1, dyn)
        self.declare_parameter("debug", False)

        gp = self.get_parameter
        self.input_topic = gp("input_topic").value
        self.doa_topic = gp("doa_topic").value
        self.output_topic = gp("output_topic").value
        self.keyword = str(gp("keyword").value).strip().lower()
        self.min_conf = float(gp("min_conf").value)
        self.require_isolated = bool(gp("require_isolated").value)
        self.refractory_s = float(gp("refractory_s").value)
        self.pair_window_s = float(gp("pair_window_s").value)
        self.chunk_s = float(gp("chunk_s").value)
        self.debug = bool(gp("debug").value)

        if Model is None:
            raise RuntimeError(
                "vosk is not importable. It is pip-installed in the dorai "
                "image; wake_mod adds no new dependency and cannot run without it.")
        SetLogLevel(-1 if not self.debug else 0)

        model_path = str(gp("model_path").value or "").strip()
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Vosk model path not found: {model_path}")
            self.model = Model(model_path)
        else:
            self.model = Model(lang=str(gp("model_lang").value))

        self._check_vocabulary()
        self._fed = 0                        # samples fed since recogniser start
        self._build_recognizer()

        self._doa = deque(maxlen=300)        # (stamp, bearing, confidence, valid)
        # See doa_mod for why this window is short and why a backwards jump
        # is handled explicitly: capture_s restarts at 0 when voice_mod does,
        # and a running minimum over stale samples would time-stamp every wake
        # event into the past -- which would then pair it with the wrong
        # bearing, or with none at all.
        self._offsets = deque(maxlen=100)    # epoch - capture_s
        self._last_cap0 = None
        self._last_fire = 0.0
        self._last_restart = 0.0
        self._n_frames = 0
        self._n_fired = 0
        self._n_segments = 0

        self.pub = self.create_publisher(String, self.output_topic, 10)
        self.create_subscription(
            Float32MultiArray, self.input_topic, self.on_audio, 10)
        self.create_subscription(String, self.doa_topic, self.on_doa, 20)
        self.create_timer(30.0, self._log_health)

        self.get_logger().info(
            f"wake_mod: listening on {self.input_topic} for '{self.keyword}' "
            f"(isolated={self.require_isolated}, min_conf={self.min_conf}), "
            f"pairing bearings from {self.doa_topic} within "
            f"{self.pair_window_s:.1f}s, publishing {self.output_topic}")

    def _check_vocabulary(self):
        """Refuse to start on an out-of-vocabulary keyword.

        Vosk only WARNS ("Ignoring word missing in vocabulary") and then builds
        a grammar that can never match, which at 3 a.m. on demo day looks
        exactly like "the wake word just does not work". `vosk_model_find_word`
        returns -1 for a word the lexicon does not have, so the failure can be
        loud and immediate instead. Wrapped because it is a thin ctypes binding
        that could move between vosk releases; if it is gone we warn and carry
        on rather than refusing to run over a missing diagnostic.
        """
        try:
            wid = self.model.vosk_model_find_word(self.keyword)
        except Exception as exc:
            self.get_logger().warning(
                f"cannot verify '{self.keyword}' against the model lexicon "
                f"({exc}); if the wake word never fires, that is the first "
                f"thing to check.")
            return
        if wid is None or int(wid) < 0:
            raise RuntimeError(
                f"keyword '{self.keyword}' is NOT in this Vosk model's "
                f"lexicon, so the grammar would contain only [unk] and could "
                f"never fire. Pick a word the model knows (e.g. marvin, "
                f"jarvis, compass), or add a lexicon entry and rebuild the "
                f"model's Gr.fst. The robot's own name 'artaban' is one of "
                f"the words this model does not have.")
        self.get_logger().info(
            f"keyword '{self.keyword}' is in the model lexicon (word id {wid}).")

    def _build_recognizer(self):
        """Grammar-constrained recogniser: the keyword, plus explicit garbage.

        `[unk]` is not decoration. Without it the decoder must map every sound
        it hears onto the keyword, because that is the only path through the
        grammar, and the wake word fires on coughs. With it, the keyword has
        to out-score a garbage model to win.

        NOTHING EVER CALLS Reset() ON THIS OBJECT. Vosk reports word times in
        seconds from the recogniser's own stream start, and that clock is what
        converts a detection into the acoustic instant the bearing pairing
        needs. As long as the recogniser is never reset, its stream time is
        exactly `self._fed / 16000` and the conversion is arithmetic. Reset()
        does not reliably return the clock to zero -- measured, it left about
        1.6 s of residual, which put every published stamp 1.6 s in the FUTURE
        and paired every wake word with no bearing at all. So when the stream
        genuinely has to restart, we build a NEW recogniser, which starts at
        zero by construction, and zero `_fed` with it.
        """
        grammar = json.dumps([self.keyword, "[unk]"])
        self.rec = KaldiRecognizer(self.model, 16000, grammar)
        self.rec.SetWords(True)
        self._fed = 0

    # -- inputs --------------------------------------------------------------
    def on_doa(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        if d.get("topic") != "doa":
            return
        try:
            self._doa.append((float(d["stamp"]), float(d.get("bearing_deg", 0.0)),
                              float(d.get("confidence", 0.0)),
                              bool(d.get("valid", False))))
        except Exception:
            return

    def on_audio(self, msg):
        """Feed one clean-audio frame through the grammar recogniser.

        Header contract: data[0:4] = [capture_s, sample_rate, num_channels,
        sequence_number], then the samples.
        """
        recv = time.time()
        data = msg.data
        if len(data) < 5:
            return
        cap0 = float(data[0])
        fs = int(data[1])
        if fs != 16000:
            self.get_logger().error(
                f"{self.input_topic} is {fs} Hz; the recogniser is 16 kHz.",
                throttle_duration_sec=10.0)
            return
        x = np.asarray(data[4:], dtype=np.float32)
        if x.size == 0:
            return
        self._n_frames += 1
        # Same clock bridge doa_mod uses: capture_s is voice_mod-relative, and
        # the minimum of (receive epoch - capture_s) is the least-delayed
        # sample and so the least-biased estimate of voice_mod's t0 in epoch.
        # Rebuilding the recogniser is expensive (the grammar FST is
        # recompiled), so the restart path is rate-limited. Two publishers on
        # the input topic at once -- which happens with a stray test rig, not
        # in the deployed pipeline -- would otherwise interleave two capture
        # clocks and rebuild on every single frame, and the node would spend
        # all its time compiling and never decode a word.
        if (self._last_cap0 is not None and cap0 < self._last_cap0 - 0.5
                and time.monotonic() - self._last_restart > 5.0):
            self._last_restart = time.monotonic()
            self.get_logger().warning(
                f"{self.input_topic} capture clock went backwards "
                f"({self._last_cap0:.1f}s -> {cap0:.1f}s): voice_mod restarted. "
                f"Resetting the clock offset and the recogniser.")
            self._offsets.clear()
            self._build_recognizer()          # also zeroes _fed
        self._last_cap0 = cap0
        self._offsets.append(recv - cap0)
        offset = min(self._offsets)

        pcm = np.clip(x, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        step = max(1, int(self.chunk_s * fs))
        for start in range(0, pcm.size, step):
            chunk = pcm[start:start + step]
            # Vosk reports word times in seconds from the recogniser's OWN
            # stream start (verified: five keywords 4 s apart came back at
            # 2.97, 6.96, 10.92, 14.91, 18.90 s). Anchoring that stream to the
            # epoch on every chunk -- rather than once, at the start -- keeps
            # the mapping correct as the clock-offset estimate settles.
            stream_t0 = (cap0 + start / float(fs) + offset
                         - self._fed / float(fs))
            done = self.rec.AcceptWaveform(chunk.tobytes())
            self._fed += chunk.size
            if not done:
                continue
            self._n_segments += 1
            try:
                words = json.loads(self.rec.Result()).get("result", [])
            except Exception:
                continue
            hit = self._match(words)
            if hit is not None:
                # The word's own `end`, not the time we noticed it: Vosk
                # endpoints about 0.8 s late and the bearing pairing needs the
                # acoustic instant, not the detection instant.
                self._fire(stream_t0 + float(hit["end"]), hit)

    def _match(self, words):
        """The endpointed result, judged. Returns the keyword entry or None.

        See the module docstring for why isolation carries this rule and a
        confidence floor alone does not.
        """
        if not words:
            return None
        hits = [w for w in words
                if w.get("word") == self.keyword
                and float(w.get("conf", 0.0)) >= self.min_conf]
        if not hits:
            return None
        if self.require_isolated and len(words) != 1:
            self.get_logger().debug(
                f"keyword seen inside a longer utterance "
                f"({[w.get('word') for w in words]}); not a wake word.")
            return None
        return hits[0]

    # -- output --------------------------------------------------------------
    def _fire(self, stamp, hit=None):
        now = time.monotonic()
        if now - self._last_fire < self.refractory_s:
            return
        self._last_fire = now
        bearing, conf = self._pair(stamp)
        payload = {
            "v": SCHEMA_V,
            "topic": "wake",
            "stamp": round(float(stamp), 3),
            "keyword": self.keyword,
            "bearing_deg": round(float(bearing), 1),
            "confidence": round(float(conf), 3),
        }
        self.pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        self._n_fired += 1
        self.get_logger().info(
            f"WAKE {json.dumps(payload)}"
            + (f"  (word conf {hit['conf']:.2f}, "
               f"{hit['end'] - hit['start']:.2f}s long)" if hit else ""))

    def _pair(self, stamp):
        """Bearing whose stamp is nearest the wake word's audio window.

        Prefer a `valid` bearing; fall back to the nearest one at all but with
        confidence 0.0, which is what DEMO_INTERFACES asks for when no
        confident bearing is available -- the consumer decides whether to turn.
        """
        near = [d for d in self._doa if abs(d[0] - stamp) <= self.pair_window_s]
        good = [d for d in near if d[3]]
        if good:
            d = min(good, key=lambda e: abs(e[0] - stamp))
            return d[1], d[2]
        if near:
            d = min(near, key=lambda e: abs(e[0] - stamp))
            return d[1], 0.0
        return 0.0, 0.0

    def _log_health(self):
        self.get_logger().info(
            f"frames={self._n_frames} endpointed_utterances={self._n_segments} "
            f"wakes={self._n_fired} doa_ring={len(self._doa)} "
            f"keyword='{self.keyword}'")


def main(args=None):
    rclpy.init(args=args)
    node = WakeMod()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
