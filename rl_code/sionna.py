import torch
import sionna.phy
from rl_code.channel_gen import ChannelGenerator
from rl_code.signal_gen import SignalGenerator

class Sionna:
    def __init__(self, *, seed: int = 42) -> None:
        sionna.phy.config.seed = seed
        self.channel_gen = ChannelGenerator()
        self.signal_gen = SignalGenerator()

    def change_seed(self, seed: int) -> None:
        sionna.phy.config.seed = seed

    def apply_channel(self, x: torch.Tensor, snr_db: float) -> tuple[torch.Tensor, torch.Tensor]:
        return self.channel_gen.apply_flatfade_channel(x, snr_db)

    def generate_signal(
        self,
        *,
        num_symbols: int,
        carrier_freq: float = 4e6,
        symbol_rate: float = 1e6,
        samples_per_symbol: int = 16,
        roll_off: float = 0.35,
        span_in_symbols: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.signal_gen.generate_16_qam(
            num_symbols=num_symbols,
            carrier_freq=carrier_freq,
            symbol_rate=symbol_rate,
            samples_per_symbol=samples_per_symbol,
            roll_off=roll_off,
            span_in_symbols=span_in_symbols,
        )



    