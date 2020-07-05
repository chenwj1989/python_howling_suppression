#!/usr/bin/python
from __future__ import division
# import

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
import pyHowling


def howling_detect(frame, win, nFFT, Slen, candidates, frame_id):
    insign = win * frame
    spec = np.fft.fft(insign, nFFT, axis=0)

    #==========  Howling Detection Stage =====================#   
    ptpr_idx = pyHowling.ptpr(spec[:Slen], 10)
    papr_idx, papr = pyHowling.papr(spec[:Slen], 10)
    pnpr_idx = pyHowling.pnpr(spec[:Slen], 15)
    intersec_idx = np.intersect1d(ptpr_idx, np.intersect1d(papr_idx,pnpr_idx))
    #print("papr:",papr_idx)
    #print("pnpr:",pnpr_idx)
    #print("intersection:", intersec_idx)
    for idx in intersec_idx:
        candidates[idx][frame_id] = 1
    ipmp = pyHowling.ipmp(candidates, frame_id)
    #print("ipmp:",ipmp)
    result = pyHowling.screening(spec, ipmp)
    #print("result:", result)
    return result

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

     # ====== set STFT parameters ========
    interval = 0.01 #frame interval = 0.02s
    Slen = int(np.floor(interval * Srate))
    if Slen % 2 == 1:
        Slen = Slen + 1
    PERC = 50  #window overlap in percent of frame size
    len1 = int(np.floor(Slen * PERC / 100))
    len2 = int(Slen - len1)
    nFFT = 2 * Slen
    freqs = np.linspace(0, Srate, nFFT)
    Nframes = int(np.floor(len(x) / len2) - np.floor(Slen / len2))
    
    #Hanning window for stft
    win = np.hanning(Slen)
    win = win * len2 / np.sum(win)

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


    #=============================Notch Filtering =======================================================     
    #                                       ___________________       
    #                             -------> | Howling Detection | ______
    #                            |         |___________________|       |
    #                            |                                     |
    #                            |      _______________         _______V______                     
    #   clean speech: x --> mic: x1 --> | Internal Gain |-x2--> | Notch Filter | --> speaker : y 
    #                        ^          |______G________|       |_____IIR______|         |          
    #                        |                                                           |
    #                        |                      _______________                      |
    #                        <-----------------y1--| Room Impulse  |____________________ v
    #                                              |____Response___|            
    #                         
    b = [1.0, 0 ,0]
    a = [0, 0, 0]
    N = min(2000, len(rir)) #limit room impulse response length
    x2 = np.zeros(100)   #
    x3 = np.zeros(N)   #buffer N samples of speaker output to generate acoustic feedback
    y = np.zeros(len(x))  #save speaker output to y 
    y1 = 0.0           #init as 0
    current_frame = np.zeros(Slen)
    pos = 0
    candidates = np.zeros([Slen, Nframes+1], dtype='int')
    frame_id = 0
    notch_freqs = []

    for i in range(len(x)):
        x1 = x[i] + y1   
        current_frame[pos] = x1
        pos = pos + 1
        if pos==Slen:
            #update notch filter frame by frame
            freq_ids = howling_detect(current_frame, win, nFFT, Slen, candidates, frame_id)
            #freq_ids = [46]
            if(len(freq_ids)>0 and (len(freq_ids)!=len(notch_freqs) or not np.all(np.equal(notch_freqs, freqs[freq_ids])))):
                notch_freqs = freqs[freq_ids]
                sos = np.zeros([len(notch_freqs), 6])
                for i in range(len(notch_freqs)):
                    b0, a0 = signal.iirnotch(notch_freqs[i], 1, Srate)
                    sos[i,:] = np.append(b0,a0)
                b, a = signal.sos2tf(sos)
            print("frame id: ", frame_id, "/", Nframes, "notch freqs:", notch_freqs)
            current_frame[:Slen-len2] = current_frame[len2:]  #shift by len2
            pos = len2
            frame_id = frame_id + 1

        x2[1:] = x2[:len(x2)-1] 
        x2[0] = G*x1
        x2[0] = min(2, x2[0])  #amplitude clipping
        x2[0] = max(-2, x2[0])      
        y[i] = np.dot(x2[:len(b)], b) - np.dot(x3[:len(a)-1], a[1:])  #IIR filter     
        y[i] = min(2, y[i])  #amplitude clipping
        y[i] = max(-2, y[i])      
        x3[1:] = x3[:N-1] 
        x3[0] =  y[i]
        y1 = np.dot(x3, rir[:N])
    
    pyHowling.plot_notch_filter(b, a, Srate)
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


