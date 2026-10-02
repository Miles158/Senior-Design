import numpy as np

def add_white_noise(signal, snr_db):

    noise = np.random.normal(0, 1, len(signal)) * 10 ** (-snr_db / 20)

    return signal + noise

def bpsk_modulate(bits, freq, samples_per_bit, sample_rate):
    """
    Modulates a binary signal using BPSK.
    
    Parameters:
        bits: numpy array of binary bits
        samples_per_bit: number of samples per bit for the numpy arrays
        freq: carrier wave freq (1/sec)
        sample_rate: sample rate (samples/sec)
        
        """

    total_samples = len(bits) * samples_per_bit
    t = np.arange(total_samples) / sample_rate

    data_wave = np.repeat(bits, samples_per_bit)
    modulation_wave = np.sin(2 * np.pi * freq * t)

    return data_wave * modulation_wave 



def bpsk_demodulate(signal, freq, samples_per_bit, sample_rate):
    """Demodulates a BPSK signal."""

    


    return 0

