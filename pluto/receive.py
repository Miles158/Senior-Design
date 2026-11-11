import adi
import matplotlib
import matplotlib.pyplot as plt

sdr = adi.Pluto(uri="ip:localhost")
data = sdr.rx()

plt.plot(data.real, label="I")
plt.plot(data.imag, label="Q")
plt.legend()
plt.savefig("pluto/recordings/rx.png")
