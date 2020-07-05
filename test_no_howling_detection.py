#!/usr/bin/python
from __future__ import division
# import

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
from pyHowling import plot_notch_filter


def main():
    input_file = "test/LDC93S6A.wav"
    howling_file = "test/added_howling.wav"
    output_file = "test/removed_howling.wav"

    #load clean speech file
    x, Srate = sf.read(input_file)

    #pre design a room impulse response
    rir = np.loadtxt('test/path.txt', delimiter='\t') 
    plt.figure()
    plt.plot(rir)

    #G : gain from mic to speaker
    G = 0.2

    # ====== set parameters ========
    interval = 0.02 #frame interval = 0.02s
    Slen = int(np.floor(interval * Srate))
    if Slen % 2 == 1:
        Slen = Slen + 1
    PERC = 50  #window overlap in percent of frame size
    len1 = int(np.floor(Slen * PERC / 100))
    len2 = int(Slen - len1)
    nFFT = 2 * Slen

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x)
    plt.xlim(0, len(x))
    plt.subplot(2,1,2)
    plt.specgram(x, NFFT=nFFT, Fs=Srate, noverlap=len2, cmap='jet')
    plt.ylim((0, 5000))
    plt.ylabel("Frquency (Hz)")
    plt.xlabel("Time (s)")

    #simulate acoustic feekback, point-by-point
    #                                    _______________                              _______________ 
    #   clean speech: x --> mic: x1 --> | Internal Gain | --> x2 -- > speaker : y--> | Room Impulse  | 
    #                        ^          |______G________|                            |____Response___|
    #                        |                                                              |
    #                         ----------------------<-----y1--------------------------------V
    #
    N = min(2000, len(rir)) #limit room impulse response length
    x2 = np.zeros(N)   #buffer N samples of speaker output to generate acoustic feedback
    y = np.zeros(len(x))  #save speaker output to y 
    y1 = 0.0           #init as 0
    for i in range(len(x)):       
        x1 = x[i] + y1   
        y[i] = G*x1
        y[i] = min(2, y[i])   #amplitude clipping
        y[i] = max(-2, y[i])         
        x2[1:] = x2[:N-1] 
        x2[0] = y[i]
        y1 = np.dot(x2, rir[:N])
   

    sf.write(howling_file, y, Srate)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(y)
    plt.xlim((0, len(y)))
    plt.subplot(2,1,2)
    plt.specgram(y, NFFT=nFFT, Fs=Srate, noverlap=len2, cmap='jet')
    plt.ylim((0, 5000))
    plt.ylabel("Frquency (Hz)")
    plt.xlabel("Time (s)")

    #notch filter
    fs = Srate  # Sample frequency (Hz)
    f0 = 603  # Frequency to be removed from signal (Hz)
    Q = 1  # Quality factor
    # Design notch filter
    b1, a1 = signal.iirnotch(f0, Q, fs)
    sos1 = np.append(b1,a1)
    #plot_notch_filter(b1, a1, fs)
    
    f0 = 1745  # Frequency to be removed from signal (Hz)
    Q = 5  # Quality factor
    # Design notch filter
    b2, a2 = signal.iirnotch(f0, Q, fs)
    sos2 = np.append(b2,a2)
    #plot_notch_filter(b2, a2, fs)

    sos = np.vstack((sos1,sos2))
    b, a = signal.sos2tf(sos)
    plot_notch_filter(b, a, fs)


    #=============================Notch Filtering =======================================================
    #                                    _______________         ______________                     
    #   clean speech: x --> mic: x1 --> | Internal Gain |-x2--> | Notch Filter | --> speaker : y 
    #                        ^          |______G________|       |_____IIR______|         |          
    #                        |                                                           |
    #                        |                      _______________                      |
    #                        <-----------------y1--| Room Impulse  |____________________ v
    #                                              |____Response___|            
    #                         

    N = min(2000, len(rir)) #limit room impulse response length
    x2 = np.zeros(len(b))   #
    x3 = np.zeros(N)   #buffer N samples of speaker output to generate acoustic feedback
    y = np.zeros(len(x))  #save speaker output to y 
    y1 = 0.0           #init as 0
    for i in range(len(x)):       
        x1 = x[i] + y1   
        x2[1:] = x2[:len(x2)-1] 
        x2[0] = G*x1
        x2[0] = min(1, x2[0])  #amplitude clipping
        x2[0] = max(-1, x2[0])      
        y[i] = np.dot(x2, b) - np.dot(x3[:len(a)-1], a[1:])  #IIR filter
        x3[1:] = x3[:N-1] 
        x3[0] =  y[i]
        y1 = np.dot(x3, rir[:N])

    xfinal = y
    sf.write(output_file, xfinal, Srate)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(xfinal)
    plt.xlim((0, len(xfinal)))
    plt.subplot(2,1,2)
    plt.specgram(xfinal, NFFT=nFFT, Fs=Srate, noverlap=len2, cmap='jet')
    plt.ylim((0, 5000))
    plt.ylabel("Frquency (Hz)")
    plt.xlabel("Time (s)")
    plt.show()


if __name__=="__main__":
    main()


