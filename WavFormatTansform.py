import numpy as np
from scipy.io import wavfile

"""
####################################################################
WAV format              Min             Max             NumPy dtype
####################################################################
32-bit floating-point   -1.0            +1.0            float32
32-bit PCM              -2147483648     +2147483647     int32
16-bit PCM              -32768          +32767          int16
8-bit PCM               0               255             uint8
#####################################################################
"""
"""
wav_f32_to_int16
* Transform a wave file from 32-bit floating-point to 16 bit PCM format
* input_dir: The input directory
* output_dir: The output directory
"""
def wav_f32_to_int16(input_dir, output_dir):
    sampling_rate, data = wavfile.read(input_dir)
    new_data = data * np.iinfo(np.int16).max
    new_data = new_data.astype(np.int16)

    wavfile.write(output_dir, sampling_rate, new_data)