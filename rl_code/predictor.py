from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class MCS:
    index: int
    bits_per_symbol: int
    code_rate: float
    beta: float
    snr_threshold_db: float

    @property
    def spectral_efficiency(self) -> float:
        return self.bits_per_symbol * self.code_rate


CQI_TABLE = (
    MCS(1, 2, 78 / 1024, 1.49, -6.7),
    MCS(2, 2, 120 / 1024, 1.53, -4.7),
    MCS(3, 2, 193 / 1024, 1.57, -2.3),
    MCS(4, 2, 308 / 1024, 1.61, 0.2),
    MCS(5, 2, 449 / 1024, 1.63, 2.4),
    MCS(6, 2, 602 / 1024, 1.65, 4.3),
    MCS(7, 4, 378 / 1024, 3.94, 5.9),
    MCS(8, 4, 490 / 1024, 4.84, 8.1),
    MCS(9, 4, 616 / 1024, 5.60, 10.3),
    MCS(10, 6, 466 / 1024, 9.52, 11.7),
    MCS(11, 6, 567 / 1024, 12.19, 14.1),
    MCS(12, 6, 666 / 1024, 15.03, 16.3),
    MCS(13, 6, 772 / 1024, 17.56, 18.7),
    MCS(14, 6, 873 / 1024, 19.79, 21.0),
    MCS(15, 6, 948 / 1024, 21.98, 22.7),
)


class Predictor:
    def __init__(self, mcs_table: tuple[MCS, ...] = CQI_TABLE) -> None:
        self.mcs_table = mcs_table

    def estimate_snr(self, x: torch.Tensor, y: torch.Tensor, samples_per_symbol: int = 1) -> torch.Tensor:
        h_hat = (x.conj() * y).sum(-1) / x.abs().pow(2).sum(-1)
        no_hat = (y - h_hat.unsqueeze(-1) * x).abs().pow(2).mean(-1)
        signal_power = x.abs().pow(2).mean(-1)
        return samples_per_symbol * h_hat.abs().pow(2) * signal_power / no_hat

    def eesm(self, snr: torch.Tensor, beta: float) -> torch.Tensor:
        snr = snr.reshape(-1)
        return -beta * (torch.logsumexp(-snr / beta, dim=0) - math.log(snr.numel()))

    def select_mcs(self, snr: torch.Tensor) -> MCS | None:
        selected = None
        for mcs in self.mcs_table:
            snr_eff_db = 10 * torch.log10(self.eesm(snr, mcs.beta)).item()
            if snr_eff_db >= mcs.snr_threshold_db:
                selected = mcs
        return selected

    def predict(self, x: torch.Tensor, y: torch.Tensor, samples_per_symbol: int = 1) -> MCS | None:
        snr = self.estimate_snr(x, y, samples_per_symbol)
        return self.select_mcs(snr)