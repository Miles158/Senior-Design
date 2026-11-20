import numpy as np

def add_white_noise(signal, snr_db, rng, signal_power=1):
    # signal_power is the TRANSMITTED power (constellations are normalized to 1).
    # don't measure it from `signal` here: after fading / path loss that would shrink the noise
    # along with the signal and cancel the fade out.
    noise_power = signal_power / 10 ** (snr_db / 10)
    if np.iscomplexobj(signal):
        # complex baseband: split noise power evenly between I and Q
        sigma = np.sqrt(noise_power / 2)
        noise = rng.normal(0, sigma, len(signal)) + 1j * rng.normal(0, sigma, len(signal))
    else:
        noise = rng.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def rayleigh_fading(signal, rng):
    """What it is: in a city or indoors, there's usually no direct line of sight. The signal reaches the receiver by bouncing off buildings, walls and cars, so many copies arrive with slightly different delays. Depending on how those copies line up, they either add up (constructive interference) or cancel out (destructive interference).

Effect: the received signal gets multiplied by a random complex gain h:

|h| sets the strength. Sometimes the signal is boosted, and sometimes it nearly vanishes (a deep fade, which can be 20–30 dB down).
The angle of h rotates the whole constellation.

Moving even half a wavelength (about 6 cm at 2.4 GHz) can take you from a strong signal to a deep fade.

Why it matters for your project: this is the core reason to adapt at all. The SNR of each frame changes randomly, so when the channel is good you can use 16QAM, and during a fade you should fall back to BPSK."""
    # all the scattered copies add up to one random complex gain h (same for the whole frame)
    h = (rng.normal(0, 1) + 1j * rng.normal(0, 1)) / np.sqrt(2)   # divide by sqrt(2) so the average |h|^2 = 1

    return signal * h, h   # return h too so the receiver can undo it

def rician_fading(signal, k_factor, rng):
    """What it is: the same as Rayleigh, except there is a direct line-of-sight path, like a rooftop link or an open field, plus the scattered reflections. The K-factor is the ratio of direct-path power to scattered power.

K = 0 → no direct path, which is pure Rayleigh (harsh).
K → ∞ → the direct path dominates, which is basically just AWGN (calm).

Effect: fades are shallower and less frequent than with Rayleigh, because the strong direct path keeps the signal from cancelling out completely.

Why it matters for your project: one knob takes you from "easy" to "hard" channels, so you can test how the model's choices change as the environment changes."""
    # k_factor is a plain ratio, not dB: 0 = pure rayleigh, 10 = strong line of sight
    direct_power = k_factor / (k_factor + 1)    # share of power in the line-of-sight path
    scatter_power = 1 / (k_factor + 1)          # share of power in the reflections

    direct = np.sqrt(direct_power)                                                         # fixed, same every frame
    scattered = np.sqrt(scatter_power / 2) * (rng.normal(0, 1) + 1j * rng.normal(0, 1))   # random, same as rayleigh
    h = direct + scattered

    return signal * h, h

def doppler(signal, speed, carrier_freq, sample_rate, rng, num_paths=20):
    """What it is: when the transmitter, the receiver, or the objects around them move, the path lengths keep changing, so the fade from #2 keeps changing too. The Doppler frequency fd = v / λ (speed divided by wavelength) tells you how fast.

Walking at 1 m/s at 2.4 GHz gives fd ≈ 8 Hz, so the channel changes slowly.
A car at 30 m/s gives fd ≈ 240 Hz, so the channel changes quickly.

The coherence time (≈ 1/fd) is roughly how long the channel stays about the same.

Effect: consecutive frames see related but drifting channels.

Why it matters for your project: in a real system, the receiver measures the channel and reports back, and then the transmitter picks a scheme. By the time the frame is sent, the channel may have changed. A plain SNR threshold always reacts to old information, but a model can learn to predict where the channel is heading. This is the strongest argument for using ML here."""
    wavelength = 3e8 / carrier_freq           # speed of light / freq (meters)
    max_doppler = speed / wavelength          # fd = v / λ (Hz)
    t = np.arange(len(signal)) / sample_rate  # time of each sample (sec)

    h = np.zeros(len(signal), dtype=complex)
    for path in range(num_paths):
        angle = np.radians(rng.uniform(0, 360))   # direction this copy arrives from
        start_phase = np.radians(rng.uniform(0, 360))
        shift = max_doppler * np.cos(angle)       # head-on copies shift by fd, side-on copies barely shift
        phase = 2 * np.pi * shift * t + start_phase
        h += np.cos(phase) + 1j * np.sin(phase)   # this copy spins at its own doppler shift

    h = h / np.sqrt(num_paths)   # keep the average |h|^2 = 1

    return signal * h, h


def offsets(signal, phase_offset_deg, freq_offset, sample_rate):
    """What it is: the transmitter and receiver each have their own oscillator (a crystal) that sets the carrier frequency. No two crystals are identical, so they're off by a few parts per million. At 2.4 GHz, a 10 ppm error is about 24 kHz. Doppler adds a little more shift on top.

Phase offset: the receiver's carrier starts at a different phase than the transmitter's, so the constellation is rotated by a fixed angle.
Frequency offset: the two carriers run at slightly different frequencies, so that rotation keeps growing, and the constellation spins over the course of a frame.

Effect: the points drift into their neighbors' decision regions. 8PSK and 16QAM have small angles between points, so they break with only a little rotation, while BPSK can handle almost 90°.

Why it matters for your project: it hurts dense schemes much more than simple ones, in a way a single SNR number doesn't show. If the model looks at the received pilots, it can learn to see this."""
    t = np.arange(len(signal)) / sample_rate   # time of each sample (sec)

    fixed_rotation = np.radians(phase_offset_deg)    # phase offset: same angle for every sample
    growing_rotation = 2 * np.pi * freq_offset * t   # freq offset: angle keeps growing over time
    angle = fixed_rotation + growing_rotation

    rotation = np.cos(angle) + 1j * np.sin(angle)
    return signal * rotation



# ---------------- satellite to ground ----------------
# satellite_pass runs once for the whole pass (one point per frame).
# the other functions run once per frame, using that frame's elevation / range / speed.


def satellite_pass(altitude_km, max_elevation_deg, time_step):
    """What it is: a LEO satellite (~500 km up) rises over the horizon, climbs to its highest point, and sets again, all in about 10 minutes. This finds where the satellite is (as seen from the ground station) at every time step of one pass.

Returns:
    t: time of each step (sec), 0 = top of the pass
    elevation: angle above the horizon (deg)
    slant_range: straight-line distance from the ground station to the satellite (m)
    radial_speed: how fast that distance is changing (m/s), + = moving away, - = coming closer

Why it matters for your project: everything else (path loss, doppler, shadowing, atmosphere) depends on these. It makes the channel change smoothly and predictably over each pass."""
    earth_radius = 6371e3                                   # meters
    orbit_radius = earth_radius + altitude_km * 1e3         # distance from earth's center to the satellite
    orbit_speed = np.sqrt(3.986e14 / orbit_radius)          # m/s, from gravity: sqrt(GM / r)
    angular_speed = orbit_speed / orbit_radius              # rad/s the satellite sweeps around earth's center

    # how far the ground station sits to the side of the satellite's path (angle at earth's center).
    # 0 = satellite passes straight overhead (max elevation 90 deg)
    max_elevation = np.radians(max_elevation_deg)
    side_angle = np.arccos(earth_radius / orbit_radius * np.cos(max_elevation)) - max_elevation

    # how far along its path the satellite is when it pops over the horizon
    horizon_angle = np.arccos(earth_radius / orbit_radius)
    rise_angle = np.arccos(np.cos(horizon_angle) / np.cos(side_angle))
    rise_time = rise_angle / angular_speed                  # sec from the horizon to the top of the pass

    t = np.arange(-rise_time, rise_time, time_step)
    along_angle = angular_speed * t                         # how far along its path the satellite is at each time

    # angle at earth's center between the ground station and the satellite
    center_angle = np.arccos(np.cos(along_angle) * np.cos(side_angle))

    # triangle of earth's center, ground station, satellite -> law of cosines for the distance
    slant_range = np.sqrt(orbit_radius ** 2 + earth_radius ** 2 - 2 * orbit_radius * earth_radius * np.cos(center_angle))

    # height of the satellite above the ground station's horizon / distance = sin(elevation)
    height_above_horizon = orbit_radius * np.cos(center_angle) - earth_radius
    elevation = np.degrees(np.arcsin(height_above_horizon / slant_range))

    radial_speed = np.gradient(slant_range, t)              # change in distance / change in time

    return t, elevation, slant_range, radial_speed


def path_loss(signal, slant_range, reference_range):
    """What it is: the signal spreads out over a bigger and bigger sphere as it travels, so the power that lands on your antenna drops with distance squared. Free-space path loss: FSPL = 20 log10(4πd / λ).

When comparing two distances, λ cancels out: extra loss = 20 log10(d / d_ref). Twice as far = 6 dB weaker.
A 500 km satellite is ~500 km away straight overhead and ~1700 km away at 10° elevation, a swing of ~10 dB over one pass.

Effect: SNR is worst near the horizon and best at the top of the pass. The snr_db you pass to add_white_noise now means "SNR when the satellite is at reference_range" (e.g. straight overhead).

Why it matters for your project: this is the main driver. The best scheme naturally goes BPSK → QPSK → 16QAM → QPSK → BPSK across the sky."""
    extra_loss_db = 20 * np.log10(slant_range / reference_range)   # how much weaker than at reference_range
    gain = 10 ** (-extra_loss_db / 20)                              # dB -> amplitude

    return signal * gain, gain


def satellite_doppler(signal, radial_speed, carrier_freq, sample_rate):
    """What it is: the satellite moves at ~7.6 km/s. Coming toward you, the received frequency is pushed up; going away, it's pulled down. At 915 MHz that's up to about ±20 kHz (vs ~8 Hz for walking). Over a pass it sweeps in an S-curve from +max to -max, changing fastest straight overhead.

Effect: the whole constellation spins, same as the frequency offset in offsets(), just much bigger.

Different from doppler(): that one is the spread of small shifts from reflections (makes the signal flicker). This one is one big shift of the whole signal.

Why it matters for your project: real ground stations predict this from the orbit and cancel most of it. Whatever is left over is what your receiver has to deal with."""
    wavelength = 3e8 / carrier_freq        # meters
    shift = -radial_speed / wavelength     # Hz. coming closer (radial_speed < 0) pushes the freq up

    # a frame is only a few ms, so the shift barely changes within one frame: treat it as fixed
    return offsets(signal, 0, shift, sample_rate), shift


def shadowed_fading(signal, elevation_deg, rng):
    """What it is: rician_fading, but how calm or harsh it is depends on elevation (the Loo model).

High elevation: clear line of sight, few reflections, so a high K (calm).
Low elevation: the signal skims along the ground through trees and buildings, so the direct path is partly blocked and reflections take over, so a low K (harsh).

Effect: extra deep fades near the horizon, right where path loss is already worst.

Why it matters for your project: two frames can have the same average SNR but need different choices. The model can learn "low elevation = be careful"."""
    # rough numbers: tune these for your environment (open field vs city)
    k_factor_db = 15 * elevation_deg / 90            # 0 dB at the horizon, 15 dB straight up
    shadow_std_db = 6 * (1 - elevation_deg / 90)     # up to 6 dB of blocking near the horizon, none straight up
    k_factor = 10 ** (k_factor_db / 10)              # dB -> plain ratio

    direct_power = k_factor / (k_factor + 1)         # share of power in the line-of-sight path
    scatter_power = 1 / (k_factor + 1)               # share of power in the reflections

    shadow_db = -abs(rng.normal(0, shadow_std_db))   # trees / buildings can only weaken the direct path
    shadow = 10 ** (shadow_db / 20)                  # dB -> amplitude

    direct = np.sqrt(direct_power) * shadow                                                 # line of sight, partly blocked
    scattered = np.sqrt(scatter_power / 2) * (rng.normal(0, 1) + 1j * rng.normal(0, 1))    # random, same as rayleigh
    h = direct + scattered

    return signal * h, h


def atmosphere(signal, elevation_deg, rain_loss_db, scintillation_db, rng):
    """What it is: near the horizon the signal crosses much more air (about 1 / sin(elevation) times as much), so atmospheric effects are worst there. Which effect matters depends on your band:

Rain fade (above ~10 GHz, Ku/Ka band): raindrops soak up and scatter the signal, 3-20+ dB drops that last minutes. Set rain_loss_db to the loss straight up (0 = clear sky, ~1-3 Ku in rain, 10+ Ka in heavy rain).
Scintillation (UHF/L/S band, where 915 MHz is): the charged upper atmosphere has patchy spots that make the signal twinkle, like a star twinkles. Set scintillation_db to how much it wobbles straight up (~0.5-2 dB is typical, worse near the equator/poles and after sunset).

Effect: extra fading that's unpredictable and has nothing to do with the pass geometry.

Why it matters for your project: the model can't predict this from the orbit alone, so it has to watch the signal."""
    # how many "layers of atmosphere" the signal crosses. stop at 5 deg so it doesn't blow up at the horizon
    air_mass = 1 / np.sin(np.radians(max(elevation_deg, 5)))

    rain_db = rain_loss_db * air_mass                                    # steady extra loss, more air = more rain to cross
    twinkle_db = rng.normal(0, scintillation_db * np.sqrt(air_mass))    # random wobble, grows with more air (sqrt is a rough rule)

    total_db = twinkle_db - rain_db
    gain = 10 ** (total_db / 20)   # dB -> amplitude

    return signal * gain, gain


def amplifier_compression(signal, backoff_db):
    """What it is: power is scarce on a satellite, so its amplifier is run close to its maximum. Near the max it squashes big samples (the output can't keep growing) and twists their phase. This uses the Saleh model, the standard one for satellite amplifiers.

backoff_db is how far below the max the amp is run: 0 dB = pushed hard (worst), 10 dB = gentle (almost no effect).

Effect: the outer points of 16QAM get pulled inward and rotated. PSK points all have the same size, so they're barely hurt.

Why it matters for your project: a real penalty on dense QAM that SNR doesn't show. It's why real satellites (DVB-S2) use APSK (rings of points) instead of square QAM."""
    drive = 10 ** (-backoff_db / 20)        # scale the input so it sits backoff_db below the max
    strength = np.abs(signal) * drive       # how big each sample is going into the amp
    angle = np.angle(signal)

    # saleh model (simplified numbers): output grows ~2x for small inputs and flattens out at strength = 1
    out_strength = 2 * strength / (1 + strength ** 2)
    twist = np.radians(60) * strength ** 2 / (1 + strength ** 2)   # bigger samples get more phase twist (up to 60 deg)

    out_strength = out_strength / (2 * drive)   # undo the 2x and the drive so small samples come out unchanged
    out_angle = angle + twist

    return out_strength * (np.cos(out_angle) + 1j * np.sin(out_angle))
