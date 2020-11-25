import numpy as np
import matplotlib.pyplot as plt

class Frame(object):
    def __init__(self, frame, framing_cache):
        self.frame = frame
        self.sampling_rate, self.fps, self.shift_rate = framing_cache
        self.fn, self.ppf, self.channel = frame.shape

    '''
    ########################### SHORT ENERGY ANALYSIS #############################
    ### MIDDLE VARIABLES:
    -> fps: every frame lasts for the time of 1/fps second
    -> fn: total number of frames
    -> shift_rate: shifted length / frame length
    ###############################################################################
    '''
    def energy_analysis(self):
        fps = self.fps
        fn = self.fn
        shift_rate = self.shift_rate
        channel = self.channel
        spf = 1 / fps
        duration = spf * fn * shift_rate
        E = np.zeros((fn, channel))
        for c in np.arange(channel):
            for f in np.arange(fn):
                E[f, c] = np.sum(self.frame[f, :, c] * self.frame[f, :, c])
        time = np.linspace(0, duration, fn)

        plt.figure()
        for cn in np.arange(channel):
            plt.subplot(1, channel, cn + 1)
            plt.xlabel('Time')
            plt.ylabel('Energy')
            plt.title('channel' + str(cn + 1))
            plt.plot(time, E[:, cn])



    '''
    #################### Short Term Zero Crossing Rate (ZCR) ##############################
    '''
    def ZCR(self):
        fps = self.fps
        fn = self.fn
        shift_rate = self.shift_rate
        channel = self.channel
        frame = self.frame
        spf = 1 / fps
        duration = spf * fn * shift_rate                # time the signal lasts
        Zn = np.zeros((fn, channel))                    # store the zero crossing rate
        for c in np.arange(channel):
            for f in np.arange(fn):
                frame[f, :, c] = np.sign(frame[f, :, c])    # do the following: positive -> 1, negative -> -1, zero -> 0
                for p in np.arange(self.ppf-1):
                    Zn[f, c] += 0.5 * (np.abs(frame[f, p+1, c] - frame[f, p, c]))
        time = np.linspace(0, duration, fn)

        plt.figure()                                    # plot the zcr graph
        for cn in np.arange(channel):
            plt.subplot(1, channel, cn + 1)
            plt.xlabel('Time')
            plt.ylabel('ZCR')
            plt.title('channel' + str(cn + 1))
            plt.plot(time, Zn[:, cn])

    '''
    #################### Auto-correlate #####################################
    fn: index of the frame
    channel
    mode: 'full' -> each side has the same number of sampling point as the
                    input frame
          'same' -> The total number of sampling point is the same as the
                    input frame
    ##########################################################################
    '''
    def auto_correlate(self, fn, channel):
        frame = self.frame[fn, :, channel]
        result = np.correlate(frame, frame, mode='full')
        plt.figure()
        plt.plot(result)

    '''
    ##################### show frame #########################
    '''
    def show(self, fn, channel):
        plt.figure()
        plt.plot(self.frame[fn, :, channel])

    def fft(self, fn, channel, min_hz=0, max_hz=None):
        if max_hz == None:
            max_hz = np.int(self.sampling_rate / 2)

        frame = self.frame[fn, :, channel]
        frame = frame / np.max(frame)   # signal normalization

        fft = np.fft.fft(frame)         # apply FFT on the input signal
        fft = np.abs(fft)               # show the results in dB form
        fft = 10 * np.log(fft / np.max(fft))
        min_point = np.int(self.ppf / self.sampling_rate * min_hz)  # specify the frequency range
        max_point = np.int(self.ppf / self.sampling_rate * max_hz)
        new_fft = fft[min_point:max_point]
        hz = np.linspace(min_hz, max_hz, max_point - min_point)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.title('Time Domain')
        plt.plot(frame)

        plt.subplot(2, 1, 2)
        plt.xlabel('kHz')
        plt.ylabel('dB')
        plt.title('Frequency Domain')
        plt.plot(hz, new_fft)

    def fft_1(self, fn, channel):
        dt = 1 / self.sampling_rate
        frame = self.frame[fn, :, channel]
        freq_vector = np.fft.rfftfreq(self.ppf, d=dt)
        X = np.fft.rfft(frame)
        log_X = np.log(np.abs(X))
        plt.figure()
        plt.plot(freq_vector, log_X)

    def cepstrum(self, fn, channel):
        dt = 1 / self.sampling_rate
        frame = self.frame[fn, :, channel]
        freq_vector = np.fft.rfftfreq(self.ppf, d=dt)
        X = np.fft.rfft(frame)
        log_X = np.log(np.abs(X))
        ceps = np.fft.rfft(log_X)
        df = freq_vector[1] - freq_vector[0]
        quefrency_vector = np.fft.rfftfreq(log_X.size, df)
        plt.figure()
        plt.plot(quefrency_vector, ceps)

    def pow_spectrum(self, fn, channel, min_hz=0, max_hz=None):
        if max_hz == None:
            max_hz = np.int(self.sampling_rate / 2)

        frame = self.frame[fn, :, channel]
        frame = frame / np.max(frame)  # signal normalization

        fft = np.fft.fft(frame)  # apply FFT on the input signal
        fft = np.abs(fft)  # show the results in dB form

        pow_spec = np.square(fft) / self.ppf
        pow_spec = 10 * np.log(fft / np.max(pow_spec))
        min_point = np.int(self.ppf / self.sampling_rate * min_hz)  # specify the frequency range
        max_point = np.int(self.ppf / self.sampling_rate * max_hz)
        new_pow = pow_spec[min_point:max_point]
        hz = np.linspace(min_hz, max_hz, max_point - min_point)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.title('Time Domain')
        plt.plot(frame)

        plt.subplot(2, 1, 2)
        plt.xlabel('kHz')
        plt.ylabel('dB')
        plt.title('Power Spectrum')
        plt.plot(hz, new_pow)

'''    def power_spectrum_co(self, fn, channel):
        frame = self.frame[fn, :, channel]
        frame = frame / np.max(frame)  # signal normalization

        auto_correlation = np.correlate(frame, frame, mode='full')
        pow_spec = np.abs(np.fft.fft(auto_correlation))
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.title('Time Domain')
        plt.plot(frame)

        plt.subplot(2, 1, 2)
        plt.xlabel('kHz')
        plt.ylabel('dB')
        plt.title('Power Spectrum')
        plt.plot(pow_spec)'''