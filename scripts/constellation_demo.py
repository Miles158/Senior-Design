import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from src import helpers
from src import constellation

rng = np.random.default_rng()
num_symbols = 100000  # symbols to send for each scheme
snrs = np.arange(0, 21)  # Es/N0 in dB


def theory_ber(scheme, snr_db):
    """Textbook BER curves to compare the simulation against."""
    snr = 10 ** (snr_db / 10)
    if scheme == "bpsk":
        return 0.5 * erfc(np.sqrt(snr))
    if scheme == "qpsk":
        return 0.5 * erfc(np.sqrt(snr / 2))
    if scheme == "8psk":
        return erfc(np.sqrt(snr) * np.sin(np.pi / 8)) / 3
    if scheme == "16qam":
        return 0.375 * erfc(np.sqrt(snr / 10))


plt.figure(figsize=(12, 6))

for scheme in constellation.CONSTELLATIONS:
    const = constellation.CONSTELLATIONS[scheme]
    num_bits = num_symbols * constellation.bits_per_symbol(const)
    bits_to_send = rng.integers(0, 2, size=num_bits)

    symbols = constellation.encode(bits_to_send, scheme)

    received_bits = constellation.demodulate(symbols, const)
    print(f"{scheme}: errors with no noise = {np.sum(bits_to_send != received_bits)} (should be 0)")

    ber_list = []
    for snr in snrs:
        noisy_symbols = helpers.add_white_noise(symbols, snr_db=snr, rng=rng)
        received_bits = constellation.demodulate(noisy_symbols, const)

        bit_errors = np.sum(bits_to_send != received_bits)
        ber = bit_errors / num_bits
        ber_list.append(ber)

        print(f"{scheme} SNR: {snr} dB, Bit Errors: {bit_errors}, BER: {ber:.6f}, Theory: {theory_ber(scheme, snr):.6f}")

    plt.semilogy(snrs, ber_list, 'o', label=scheme)  
    plt.semilogy(snrs, theory_ber(scheme, snrs), 'k--')

plt.title('BER vs SNR (dots = simulation, dashed = theory)')
plt.xlabel('Es/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.ylim(1e-6, 1)
plt.legend()
plt.grid()
plt.savefig('plots/ber_all_schemes.png')