import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from src import channel
from src import constellation

rng = np.random.default_rng()
num_symbols = 100000         # symbols for the AWGN test
num_frames = 2000            # frames for the fading test (new h every frame)
frame_len = 100              # symbols per frame
snrs = np.arange(0, 31, 2)   # Es/N0 in dB


# ---------------- Test 1: AWGN adds exactly the noise we ask for ----------------
bits_to_send = rng.integers(0, 2, size=num_symbols * 2)
symbols = constellation.encode(bits_to_send, "qpsk")


# ---------------- Test 2: rayleigh fading BER should match theory ----------------
const = constellation.CONSTELLATIONS["bpsk"]
ber_list = []

for snr in snrs:
    bit_errors = 0
    for frame in range(num_frames):
        bits_to_send = rng.integers(0, 2, size=frame_len)
        symbols = constellation.encode(bits_to_send, "bpsk")

        faded_symbols, h = channel.rayleigh_fading(symbols, rng)
        noisy_symbols = channel.add_white_noise(faded_symbols, snr_db=snr, rng=rng)
        received_bits = constellation.demodulate(noisy_symbols / h, const)   # undo the fade with the true h

        bit_errors += np.sum(bits_to_send != received_bits)

    ber = bit_errors / (num_frames * frame_len)
    ber_list.append(ber)
    print(f"Rayleigh BPSK SNR: {snr} dB, BER: {ber:.6f}")

snr_linear = 10 ** (snrs / 10)
awgn_theory = 0.5 * erfc(np.sqrt(snr_linear))
rayleigh_theory = 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))

plt.figure(figsize=(12, 6))
plt.semilogy(snrs, ber_list, 'o', label='rayleigh (simulation)')
plt.semilogy(snrs, rayleigh_theory, 'k--', label='rayleigh (theory)')
plt.semilogy(snrs, awgn_theory, 'k:', label='no fading (theory)')
plt.title('BPSK BER with Rayleigh fading')
plt.xlabel('Average Es/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.ylim(1e-6, 1)
plt.legend()
plt.grid()
plt.savefig('plots/ber_rayleigh.png')


# ---------------- Test 3: loss and BER over one satellite pass ----------------
t, elevation, slant_range, radial_speed = channel.satellite_pass(500, 60, time_step=5)
snr_closest = 20             # SNR (dB) when the satellite is closest
symbols_per_frame = 2000

path_loss_list = []
total_loss_list = []
ber_lists = {scheme: [] for scheme in constellation.CONSTELLATIONS}

for i in range(len(t)):
    # pass in a signal of 1 just to get back each gain
    _, path_gain = channel.path_loss(1, slant_range[i], slant_range.min())
    _, air_gain = channel.atmosphere(1, elevation[i], rain_loss_db=0, scintillation_db=0.5, rng=rng)
    _, h = channel.shadowed_fading(1, elevation[i], rng)
    total_gain = path_gain * air_gain * h

    path_loss_list.append(-20 * np.log10(path_gain))
    total_loss_list.append(-20 * np.log10(np.abs(total_gain)))

    # every scheme sees the same channel in this frame
    for scheme in constellation.CONSTELLATIONS:
        const = constellation.CONSTELLATIONS[scheme]
        bits_to_send = rng.integers(0, 2, size=symbols_per_frame * constellation.bits_per_symbol(const))
        symbols = constellation.encode(bits_to_send, scheme)

        noisy_symbols = channel.add_white_noise(symbols * total_gain, snr_db=snr_closest, rng=rng)
        received_bits = constellation.demodulate(noisy_symbols / total_gain, const)   # undo the loss with the true gain

        ber_lists[scheme].append(np.mean(bits_to_send != received_bits))

plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.plot(t, elevation)
plt.title('One satellite pass (500 km, 60 deg max elevation)')
plt.ylabel('Elevation (deg)')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, path_loss_list, label='path loss only')
plt.plot(t, total_loss_list, '.', label='path loss + atmosphere + shadowing')
plt.ylabel('Loss vs closest (dB)')
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
for scheme in ber_lists:
    plt.semilogy(t, ber_lists[scheme], '.', label=scheme)   # frames with 0 errors can't show on a log scale
plt.xlabel('Time from top of pass (s)')
plt.ylabel('Bit Error Rate (BER)')
plt.ylim(1e-8, 1)
plt.legend()
plt.grid()
plt.savefig('plots/satellite_pass.png')


# ---------------- Test 4: what noise, fading and the amplifier do to the constellation ----------------
# red + = where the points should be.
plot_rng = np.random.default_rng(3)   # fixed seed so this figure looks the same every run
num_plot_symbols = 3000


def draw_ideal(const):
    """Red + where the points should be, and the same axes on every panel."""
    plt.scatter(const.real, const.imag, color='red', marker='+', s=60, zorder=3)
    plt.xlim(-1.8, 1.8)
    plt.ylim(-1.8, 1.8)
    plt.gca().set_aspect('equal')
    plt.grid(alpha=0.4)


plt.figure(figsize=(13, 8.5))

for row, scheme in enumerate(["qpsk", "16qam"]):
    const = constellation.CONSTELLATIONS[scheme]
    bits_to_send = plot_rng.integers(0, 2, size=num_plot_symbols * constellation.bits_per_symbol(const))
    symbols = constellation.encode(bits_to_send, scheme)

    # 1. AWGN: every point becomes a fuzzy cloud
    plt.subplot(2, 3, row * 3 + 1)
    noisy_symbols = channel.add_white_noise(symbols, snr_db=16, rng=plot_rng)
    plt.scatter(noisy_symbols.real, noisy_symbols.imag, s=3, alpha=0.3)
    draw_ideal(const)
    plt.ylabel(f'{scheme.upper()}\nQuadrature (Q)', fontweight='bold')
    if row == 0:
        plt.title('AWGN')

    # 2. rayleigh fading: each frame gets its own h, which shrinks/grows AND rotates the whole constellation
    plt.subplot(2, 3, row * 3 + 2)
    fade_rng = np.random.default_rng(48458)   # same 2 mild fades for both schemes (one a bit stronger, one a bit weaker)
    for frame in range(2):
        faded_points, h = channel.rayleigh_fading(const, fade_rng)   # just the ideal points (no noise), so each one moves exactly once
        label = f'frame {frame + 1}: {20 * np.log10(abs(h)):+.0f} dB, {np.degrees(np.angle(h)):+.0f}°'
        plt.scatter(faded_points.real, faded_points.imag, s=40, label=label)
    draw_ideal(const)
    plt.legend(fontsize=8, loc='upper left')
    if row == 0:
        plt.title('Rayleigh fading')

    # 3. amplifier compression: big points get squashed and twisted a little more than small ones
    plt.subplot(2, 3, row * 3 + 3)
    amp_points = channel.amplifier_compression(const, backoff_db=7)   # just the ideal points (no noise), 7 dB below the amp's max: gentle
    # a receiver with pilots fixes one overall gain + rotation. whatever is left over is the real damage
    overall = np.sum(amp_points * np.conj(const)) / np.sum(np.abs(const) ** 2)
    amp_points = amp_points / overall
    plt.scatter(amp_points.real, amp_points.imag, s=40, color='C2')
    draw_ideal(const)
    if row == 0:
        plt.title('Amplifier compression with Saleh Model')

for panel in range(4, 7):
    plt.subplot(2, 3, panel)
    plt.xlabel('In-Phase (I)')

plt.suptitle('Demonstrations of Noise Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/constellation_impairments.png', dpi=150)



# ---------------- Test 5: doppler over a satellite pass ----------------
# coming toward you the freq is pushed up, straight overhead it's 0, going away it's pulled down.
# the receiver cancels most of it using the orbit. whatever is left makes the constellation slowly spin.
carrier_freq = 915e6
symbol_rate = 100e3                  # symbols/sec
frame_symbols = 3000                 # 3000 symbols at 100k/sec = a 30 ms frame
leftover = 0.00015                   # pretend the orbit-based correction removes all but 0.015% of the shift

t, elevation, slant_range, radial_speed = channel.satellite_pass(500, 60, time_step=1)
_, doppler_shift = channel.satellite_doppler(np.ones(1), radial_speed, carrier_freq, symbol_rate)   # dummy signal, just to get the shifts back

const = constellation.CONSTELLATIONS["qpsk"]
bits_to_send = plot_rng.integers(0, 2, size=frame_symbols * constellation.bits_per_symbol(const))
symbols = constellation.encode(bits_to_send, "qpsk")

leftover_shift = doppler_shift.max() * leftover   # leftover at the biggest shift (start of the pass)
spinning_symbols = channel.offsets(symbols, phase_offset_deg=0, freq_offset=leftover_shift, sample_rate=symbol_rate)
spinning_symbols = channel.add_white_noise(spinning_symbols, snr_db=20, rng=plot_rng)
time_ms = np.arange(frame_symbols) / symbol_rate * 1e3

plt.figure(figsize=(12, 10))

# top: the doppler S-curve over the whole pass
plt.subplot(2, 1, 1)
plt.plot(t, doppler_shift / 1e3)
plt.title('Doppler effect with a simulated satellite pass (500 km, 915 MHz)')
plt.xlabel('Time from top of pass (s)')
plt.ylabel('Doppler shift (kHz)')

# bottom left: plain QPSK
plt.subplot(2, 2, 3)
plt.scatter(symbols.real, symbols.imag, s=40)
draw_ideal(const)
plt.xlim(-1.4, 1.4)
plt.ylim(-1.4, 1.4)
plt.title('Plain QPSK')
plt.xlabel('In-Phase (i)')
plt.ylabel('Quadrature (q)')

# bottom right: AWGN + leftover doppler, colored by time so you can see it spin
plt.subplot(2, 2, 4)
plt.scatter(spinning_symbols.real, spinning_symbols.imag, c=time_ms, cmap='viridis', s=3)
draw_ideal(const)
plt.xlim(-1.4, 1.4)
plt.ylim(-1.4, 1.4)
plt.title(f'I-Q plane with AWGN and doppler effect over a small time interval')
plt.xlabel('In-Phase (i)')
plt.ylabel('Quadrature (q)')

plt.tight_layout()
plt.savefig('plots/doppler_pass.png', dpi=150)
