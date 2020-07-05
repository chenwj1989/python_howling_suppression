
import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal
import pyroomacoustics as pra

def plot_stft(x, fs, Slen, len2, nFFT):
    
    Nframes = int(np.floor(len(x) / len2) - np.floor(Slen / len2))
    #Hanning window for stft
    win = np.hanning(Slen)
    win = win * len2 / np.sum(win)

    specs = np.zeros([Slen, Nframes+1])
    for i in range(Nframes):
        k = i*len2
        insign = win * x[k:k + Slen]
        spec = np.fft.fft(insign, nFFT, axis=0)
        specs[:,i] = 10*np.log10(np.abs(spec[:Slen])**2)
    
    plt.figure()
    plt.imshow(specs, origin='lower', cmap='jet')
    ylocs = np.linspace(0, Slen, 5, dtype='int16')
    del_freq = fs / nFFT
    plt.yticks(ylocs, ["%0.0f" % l for l in (ylocs * del_freq)])
    plt.ylabel("frquency (Hz)")

    xlocs = np.linspace(0, Nframes, 5)
    frame_dur = 1 / float(fs) * len2
    plt.xticks(xlocs, ["%.02f" % l for l in (xlocs * frame_dur)])
    plt.xlabel("time (s)")


def plot_notch_filter(b, a, fs):
    # Frequency response
    freq, h = signal.freqz(b, a, fs=fs)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
    ax[0].set_title("Frequency Response")
    ax[0].set_ylabel("Amplitude (dB)", color='blue')
    ax[0].set_xlim([0, 5000])
    ax[0].set_ylim([-100, 10])
    ax[0].grid()
    ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    ax[1].set_ylabel("Angle (degrees)", color='green')
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_xlim([0, 100])
    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_ylim([-90, 90])
    ax[1].grid()


def genRIR(audio, fs):

    #Room Impulse Response
    # room dimension
    room_dim = [10, 10, 10] #meters

    # Create the shoebox
    room = pra.ShoeBox(
        room_dim,
        absorption=0.1,
        fs=fs,
        max_order=15,
        )
    # source and mic locations
    room.add_source([1, 1.05, 1], signal=audio)
    room.add_microphone_array(
            pra.MicrophoneArray(
                np.array([[1, 1, 1]]).T, 
                room.fs)
            )

    room.compute_rir()
    return room.rir[0][0]