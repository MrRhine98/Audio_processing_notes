import MyPack as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from Signal import Signal
from Frame import Frame
root = 'audio_files'
name_in = 'C_guitar.wav'
name_out = 'C_guitar2.wav'
dir_input = os.path.join(root, name_in)
dir_output = os.path.join(root, name_out)

signal = Signal(dir_input)
signal.f32_to_int16()
signal.show()

f, f_cache = signal.framing(fps=20)
frame = Frame(f, f_cache)
frame.show(0, 0)
frame.fft_1(0, 0)
frame.cepstrum(0, 0)
'''frame.energy_analysis()
frame.fft(20, 0, 0, 20000)
frame.auto_correlate(20, 0)'''

plt.show()
