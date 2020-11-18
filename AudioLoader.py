import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
from WavFormatTansform import wav_f32_to_int16
import MyPack as mp


root = 'audio_files'
name_in = 'intro.wav'
name_out = 'intro2.wav'
dir_input = os.path.join(root, name_in)
dir_output = os.path.join(root, name_out)

wav_f32_to_int16(dir_input, dir_output)
sampling_rate, data = wavfile.read(dir_output)

channel = data.shape[1]
length = data.shape[0] / sampling_rate
print('sampling_rate is:', sampling_rate)
print('number of channels:', channel)
print('Length of the file:', length)
time = np.linspace(0, length, data.shape[0])


N = np.int(sampling_rate/100)
mask = list(np.arange(N))
test_data = data[:, 0]
test_data = test_data[mask]
mp.show_fft(test_data, sampling_rate, min_hz=0, max_hz=22050)


w = mp.window('Hamming', N)
result = test_data * w
mp.show_fft(result, sampling_rate, min_hz=0, max_hz=22050)

plt.show()




