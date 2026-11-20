import torch
from sionna.phy.mapping import QAMSource
from sionna.phy.signal import Upsampling, RootRaisedCosineFilter

class SignalGenerator:
    def __init__(self):
        None

    def generate_16_qam(
        self,
        *,
        num_symbols: int,
        carrier_freq: float = 4e6,
        symbol_rate: float = 1e6,
        samples_per_symbol: int = 16,
        roll_off: float = 0.35,
        span_in_symbols: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = QAMSource(num_bits_per_symbol=4)
        upsampler = Upsampling(samples_per_symbol)
        rrc_filter = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, roll_off)

        symbols = source([1, num_symbols])
        upsampled = upsampler(symbols)
        s_bb = rrc_filter(upsampled)
        s_bb = s_bb.reshape(-1)

        fs = samples_per_symbol * symbol_rate
        t = torch.arange(s_bb.shape[0], device=s_bb.device) / fs
        carrier = torch.exp(2j * torch.pi * carrier_freq * t)
        s_pb = torch.real(s_bb * carrier)
        return t, s_bb, s_pb

    

