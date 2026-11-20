import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import sionna.phy
from sionna.phy.channel import GenerateFlatFadingChannel, AWGN

class ChannelGenerator:
    def __init__(self) -> None:
        self.flatfade_channel = GenerateFlatFadingChannel(num_tx_ant=1, num_rx_ant=1)
        self.awgn = AWGN()

    def generate_flatfade_channel(self) -> torch.Tensor:
        h = self.flatfade_channel(batch_size=1)
        h = h.reshape(())
        return h

    def apply_flatfade_channel(self, x: torch.Tensor, snr_db: float
                                                            ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.generate_flatfade_channel()
        no = x.abs().pow(2).mean() / 10 ** (snr_db / 10)
        y = self.awgn(h * x, no)
        return y, h

    
    

