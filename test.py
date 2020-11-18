import MyPack as mp
import numpy as np
import matplotlib.pyplot as plt

w1 = mp.window('Hanning', 441)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w1)
plt.subplot(2, 1, 2)
w1_fft = np.fft.fft(w1)
print(w1_fft)
w1_fft_ap = np.abs(w1_fft)
plt.plot(w1_fft_ap)
plt.show()
