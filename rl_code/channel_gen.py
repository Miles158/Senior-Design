import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import sionna.phy
from sionna.phy.channel import RayleighBlockFading

class ChannelGenerator:
    def __init__(self, *, seed: int = 42) -> None:
        sionna.phy.config.seed = seed

    def change_seed(self, seed: int) -> None:
        sionna.phy.config.seed = seed

    def 