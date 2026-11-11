import numpy as np



def bpsk_modulate(symbols, freq, samples_per_bit, sample_rate):
    """
    Modulates a binary signal using BPSK.
    
    Parameters:
        symbols: numpy array of binary symbols (-1 or 1)
        samples_per_bit: number of samples per bit for the numpy arrays
        freq: carrier wave freq (1/sec)
        sample_rate: sample rate (samples/sec)
        
        """

    total_samples = len(symbols) * samples_per_bit
    t = np.arange(total_samples) / sample_rate

    data_wave = np.repeat(symbols, samples_per_bit)
    modulation_wave = np.sin(2 * np.pi * freq * t)

    return data_wave * modulation_wave 



def bpsk_demodulate(signal, freq, samples_per_bit, sample_rate):
    """Demodulates a BPSK signal."""
    
    t = np.arange(len(signal)) / sample_rate
    carrier_wave = np.sin(2 * np.pi * freq * t)
    mixed_signal = signal * carrier_wave
    

    num_bits = len(signal) // samples_per_bit
    reshaped_signal = mixed_signal.reshape((num_bits, samples_per_bit))
    
    integrated_signal = np.sum(reshaped_signal, axis=1)
    
    symbols = (integrated_signal > 0).astype(int) * 2 - 1  
    return symbols
