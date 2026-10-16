import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from src import helpers

sample_rate = 10000  # samples per second
samples_per_bit = 100  # number of samples per bit
freq = 1000  # carrier frequency in Hz
rng = np.random.default_rng()

symbols_to_send = rng.integers(0, 2, size=100000) * 2 - 1  # generate a vector of random symbols (-1 or 1)
print(symbols_to_send)


modulated_signal = helpers.bpsk_modulate(symbols_to_send, freq=freq, samples_per_bit=samples_per_bit, sample_rate=sample_rate)
snrs = np.linspace(-50, 0, 15)  

bit_errors_list = []
ber_list = []

for snr in snrs:
    noisy_signal = helpers.add_white_noise(modulated_signal, snr_db=snr)

    demodulated_symbols = helpers.bpsk_demodulate(noisy_signal, freq=freq, samples_per_bit=samples_per_bit, sample_rate=sample_rate)
    
    bit_errors = np.sum(symbols_to_send != demodulated_symbols)
    ber = bit_errors / len(symbols_to_send)


    bit_errors_list.append(bit_errors)
    ber_list.append(ber)

    
    print(f"SNR: {snr:.2f} dB, Bit Errors: {bit_errors}, BER: {ber:.4f}")


plt.figure(figsize=(12, 6))
plt.semilogy(snrs, ber_list, marker='o')
plt.title('Bit Error Rate (BER) vs Signal-to-Noise Ratio (SNR)')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid()
plt.savefig('plots/ber_vs_snr.png')