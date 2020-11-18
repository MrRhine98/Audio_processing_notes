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
    elif name == 'Rect':
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





