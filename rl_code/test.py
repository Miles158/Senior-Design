import math

from rl_code.predictor import Predictor
from rl_code.sionna import Sionna

MODULATIONS = {2: "QPSK", 4: "16-QAM", 6: "64-QAM"}


def main():
    seed = 42
    samples_per_symbol = 10
    snr_db_values = range(-20, 30, 5)

    sionna_obj = Sionna(seed=seed)
    predictor = Predictor()
    t, s_bb, s_pb = sionna_obj.generate_signal(num_symbols=100, samples_per_symbol=samples_per_symbol)

    print(f"{'trial':>5} {'SNR [dB]':>9} {'|h|^2 [dB]':>11} {'est. Es/N0 [dB]':>16} {'MCS':>4} {'modulation':>11} {'rate':>6}")
    for trial, snr_db in enumerate(snr_db_values, start=1):
        sionna_obj.change_seed(seed)
        y, h = sionna_obj.apply_channel(s_bb, snr_db)

        snr = predictor.estimate_snr(s_bb, y, samples_per_symbol)
        mcs = predictor.select_mcs(snr)

        gain_db = 10 * math.log10(h.abs().pow(2).item())
        snr_est_db = 10 * math.log10(snr.item())

        if mcs is None:
            print(f"{trial:>5} {snr_db:>9} {gain_db:>11.1f} {snr_est_db:>16.1f} {'-':>4} {'outage':>11} {'-':>6}")
        else:
            modulation = MODULATIONS[mcs.bits_per_symbol]
            print(f"{trial:>5} {snr_db:>9} {gain_db:>11.1f} {snr_est_db:>16.1f} {mcs.index:>4} {modulation:>11} {mcs.code_rate:>6.3f}")


if __name__ == "__main__":
    main()