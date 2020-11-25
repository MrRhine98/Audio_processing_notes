import numpy as np
import matplotlib.pyplot as plt

'''
########### window function ###########
-> name: Specify the window type
-> N: The number of points in one frame
#######################################
'''
def window(name='Hamming', N=20):
    if name == 'Hamming':
        win = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hanning':
        win = np.array([0.5 * (1 - np.cos(2 * np.pi * n / (N - 1))) for n in range(N)])
    else:
        win = np.ones(N)

    return win


'''
######################## show_fft funtion ############################
-> signal: 1-D original signal in numpy.array 
-> sampling_rate
-> min_hz, max_hz: The frequency range that you want to see in the plot
#######################################################################
'''
def show_fft(signal, sampling_rate, min_hz=0, max_hz=None):
    signal = signal / np.max(signal)                        # signal normalization

    fft = np.fft.fft(signal)                                # apply FFT on the input signal
    fft = np.abs(fft)                                       # show the results in dB form
    fft = 10 * np.log(fft/np.max(fft))
    n = signal.shape[0]
    min_point = np.int(n / sampling_rate * min_hz)          # specify the frequency range
    max_point = np.int(n / sampling_rate * max_hz)
    new_fft = fft[min_point:max_point]
    hz = np.linspace(min_hz, max_hz, max_point-min_point)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.title('Time Domain')
    plt.plot(signal)

    plt.subplot(2, 1, 2)
    plt.xlabel('kHz')
    plt.ylabel('dB')
    plt.title('Frequency Domain')
    plt.plot(hz, new_fft)

'''
########################### signal_framing ####################################
###input:
-> signal: input audio signal with 1-D or higher dimension in the form of 
            np.array[sample_points, channels]
-> sampling_rate
-> fps: every frame lasts for the time of 1/fps second
-> shift_rate: shifted length / frame length

###output:
-> data: np.array[index_frame, sample_points, channel]
###############################################################################
'''
def signal_framing(signal, sampling_rate, fps=100, shift_rate=0.4, win_type='Hanning'):
    length, channel = signal.shape
    ppf = np.int(sampling_rate / fps)
    shift = np.int(ppf * shift_rate)
    num_frame = np.int((length - ppf) / shift + 1)
    data = np.zeros((num_frame, ppf, channel))

    w = window(win_type, ppf)
    for c in range(channel):
        for f in range(num_frame):
            data[f, :, c] = w * signal[f*shift:f*shift+ppf, c]
    return data


def energy_analysis(frames, fps=100, shift_rate=0.4):
    fn, ppf, channel = frames.shape
    spf = 1 / fps
    length = spf * fn * shift_rate
    E = np.zeros((fn, channel))
    for c in np.arange(channel):
        for f in np.arange(fn):
            E[f, c] = np.sum(frames[f, :, c]*frames[f, :, c])
    time = np.linspace(0, length, fn)

    plt.figure()
    for cn in np.arange(channel):
        plt.subplot(1, channel, cn+1)
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('channel'+str(cn+1))
        plt.plot(time, E[:, cn])





