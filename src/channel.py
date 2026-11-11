def add_white_noise(signal, snr_db, rng):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / 10 ** (snr_db / 10)
    if np.iscomplexobj(signal):
        # complex baseband: split noise power evenly between I and Q
        sigma = np.sqrt(noise_power / 2)
        noise = rng.normal(0, sigma, len(signal)) + 1j * rng.normal(0, sigma, len(signal))
    else:
        noise = rng.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def rayleigh_fading():
    """What it is: in a city or indoors, there's usually no direct line of sight. The signal reaches the receiver by bouncing off buildings, walls and cars, so many copies arrive with slightly different delays. Depending on how those copies line up, they either add up (constructive interference) or cancel out (destructive interference).

Effect: the received signal gets multiplied by a random complex gain h:

|h| sets the strength. Sometimes the signal is boosted, and sometimes it nearly vanishes (a deep fade, which can be 20–30 dB down).
The angle of h rotates the whole constellation.

Moving even half a wavelength (about 6 cm at 2.4 GHz) can take you from a strong signal to a deep fade.

Why it matters for your project: this is the core reason to adapt at all. The SNR of each frame changes randomly, so when the channel is good you can use 16QAM, and during a fade you should fall back to BPSK."""
    return 0

def rician_fading():
    """What it is: the same as Rayleigh, except there is a direct line-of-sight path, like a rooftop link or an open field, plus the scattered reflections. The K-factor is the ratio of direct-path power to scattered power.

K = 0 → no direct path, which is pure Rayleigh (harsh).
K → ∞ → the direct path dominates, which is basically just AWGN (calm).

Effect: fades are shallower and less frequent than with Rayleigh, because the strong direct path keeps the signal from cancelling out completely.

Why it matters for your project: one knob takes you from "easy" to "hard" channels, so you can test how the model's choices change as the environment changes."""
    return 0

def doppler():
    """What it is: when the transmitter, the receiver, or the objects around them move, the path lengths keep changing, so the fade from #2 keeps changing too. The Doppler frequency fd = v / λ (speed divided by wavelength) tells you how fast.

Walking at 1 m/s at 2.4 GHz gives fd ≈ 8 Hz, so the channel changes slowly.
A car at 30 m/s gives fd ≈ 240 Hz, so the channel changes quickly.

The coherence time (≈ 1/fd) is roughly how long the channel stays about the same.

Effect: consecutive frames see related but drifting channels.

Why it matters for your project: in a real system, the receiver measures the channel and reports back, and then the transmitter picks a scheme. By the time the frame is sent, the channel may have changed. A plain SNR threshold always reacts to old information, but a model can learn to predict where the channel is heading. This is the strongest argument for using ML here."""
    return 0


def offsets():
    """What it is: the transmitter and receiver each have their own oscillator (a crystal) that sets the carrier frequency. No two crystals are identical, so they're off by a few parts per million. At 2.4 GHz, a 10 ppm error is about 24 kHz. Doppler adds a little more shift on top.

Phase offset: the receiver's carrier starts at a different phase than the transmitter's, so the constellation is rotated by a fixed angle.
Frequency offset: the two carriers run at slightly different frequencies, so that rotation keeps growing, and the constellation spins over the course of a frame.

Effect: the points drift into their neighbors' decision regions. 8PSK and 16QAM have small angles between points, so they break with only a little rotation, while BPSK can handle almost 90°.

Why it matters for your project: it hurts dense schemes much more than simple ones, in a way a single SNR number doesn't show. If the model looks at the received pilots, it can learn to see this."""
    return 0