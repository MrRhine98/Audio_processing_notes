import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

class Signal(object):
    def __init__(self, data_dir):
        self.sampling_rate, self.data = wavfile.read(data_dir)
        if len(self.data.shape) == 1:
            self.channel = 1
            self.data = self.data.reshape(-1, 1)
        else:
            self.channel = self.data.shape[1]
        self.length = self.data.shape[0]
        self.duration = self.length / self.sampling_rate

    '''
    ########################### SIGNAL_FRAMING ####################################
    ### INPUT:
    -> signal: input audio signal with 1-D or higher dimension in the form of 
                np.array[sample_points, channels]
    -> sampling_rate
    -> fps: every frame lasts for the time of 1/fps second
    -> shift_rate: shifted length / frame length
    ### MIDDLE VARIABLES:
    -> ppf: points per frame
    -> shift: length of window shifting
    -> num_frame: number of total frames
    ### OUTPUT:
    -> data: np.array[index_frame, sample_points, channel]
    -> framing_cache: data needed for frame process
    ###############################################################################
    '''
    def framing(self, fps=100, shift_rate=0.4, win_type='Hanning'):
        ppf = np.int(self.sampling_rate / fps)
        shift = np.int(ppf * shift_rate)
        num_frame = np.int((self.length - ppf) / shift + 1)
        frame = np.zeros((num_frame, ppf, self.channel))

        w = self.window(win_type, ppf)
        for c in range(self.channel):
            for f in range(num_frame):
                frame[f, :, c] = w * self.data[f * shift:f * shift + ppf, c]
        framing_cache = (self.sampling_rate, fps, shift_rate)
        return frame, framing_cache

    def show(self):
        time = np.linspace(0, self.duration, self.length)
        plt.figure()
        plt.title('Time domain')
        for c in np.arange(self.channel):
            plt.xlabel('Time')
            plt.subplot(1, self.channel, c+1)
            plt.plot(time, self.data[:, c])

    def add_noise(self, weight=0.01):
        noise = weight * np.random.randn(self.length, self.channel) * np.iinfo(self.format()).max
        noise = np.int16(noise)
        self.data += noise


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
    -> F32_TO_INT16
    * Transform a wave file from 32-bit floating-point to 16 bit PCM format
    * input_dir: The input directory
    * output_dir: The output directory
    """
    def f32_to_int16(self, new_path=None, overwrite=False):
        new_data = self.data * np.iinfo(np.int16).max
        self.data = new_data.astype(np.int16)
        if overwrite:
            wavfile.write(new_path, self.sampling_rate, self.data)

    def format(self):
        return self.data.dtype

    '''
    ########### window function ###########
    -> name: Specify the window type
    -> N: The number of points in one frame
    #######################################
    '''
    def window(self, name='Hamming', N=20):
        if name == 'Hamming':
            win = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
        elif name == 'Hanning':
            win = np.array([0.5 * (1 - np.cos(2 * np.pi * n / (N - 1))) for n in range(N)])
        else:
            win = np.ones(N)

        return win