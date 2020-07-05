#!/usr/bin/python
from __future__ import division
# import

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display

from pySE.ns_process import NoiseSuppression
from pySE.suppression_gain import ns_gain
from pySE.compute_snr import compute_snr
from pySE.noise_estimation import noise_estimation
from pySE.speech_precense_probability import spp

def main():
    input_file = "test/sp04_babble_sn10.wav"
    output_file = "test/out.wav"
    #SAMPLE_RATE = 44100
    x, Srate = sf.read(input_file)

    # ====== Make window ========
    interval = 0.02 #frame interval = 0.02s
    Slen = int(np.floor(interval * Srate))
    if Slen % 2 == 1:
        Slen = Slen + 1
    PERC = 50  #window overlap in percent of frame size
    len1 = int(np.floor(Slen * PERC / 100))
    len2 = int(Slen - len1)

    win = np.hanning(Slen)
    win = win * len2 / np.sum(win)

    #Noise magnitude calculations - assuming that the first 6 frames is  noise/silence

    nFFT = 2 * Slen
    noise_mean =np.zeros(nFFT)

    librosa.output.write_wav(output_file, x, Srate)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(x)), ref=np.max)
    librosa.display.specshow(D, y_axis='log')
    #plt.specgram(x, NFFT=Slen, Fs=Srate, noverlap=len2, cmap='jet')
    #plt.ylim((0, 4000))
    plt.show()

    j=1
    noise_frames = 6
    for m in range(1, noise_frames, 1):
        noise_mean = noise_mean + np.absolute(np.fft.fft(win * x[j:j + Slen], nFFT, axis=0))
        j=j + Slen
    noise_mu2 = (noise_mean / noise_frames) ** 2

    #initialize
    x_old = np.zeros(len1)
    Nframes = int(np.floor(len(x) / len2) - np.floor(Slen / len2))
    xfinal = np.zeros(Nframes * len2)

    #parameters
    #aa = 0.98
    #mu = 0.98
    #eta = 0.15
    #ksi_min = 10 ** (-25 / 10)

    ns = NoiseSuppression('logmmse', 'vad', len2, Slen, nFFT)
    ns_parameters = ns.get_parameters()
    gain_parameters = ns_parameters["gain_parameters"]
    gain_parameters['gain_method'] = 'logmmse'

    noise_parameters = ns_parameters["noise_parameters"]
    noise_parameters['alpha'] = 0.98
    noise_method = 'vad'

    spp_parameters = ns_parameters["spp_parameters"]
    spp_parameters['sap_priori_method'] = 'cohen'

    snr_parameters = ns_parameters["snr_parameters"]
    snr_parameters['gamma_max'] = 40
    snr_parameters['ksi_min'] = 10 ** (-25 / 10)
    snr_parameters['dd_nr_snr'] = 0.98

    prev_ksi = []
    qk = np.ones(nFFT) * 0.5
    #=============================  Start Processing =======================================================
    for k in range(0, Nframes*len2, len2):
        insign = win * x[k:k + Slen]
        #print("k=", k)
        spec = np.fft.fft(insign, nFFT, axis=0)
        sig = np.absolute(spec)
        sig2 = sig ** 2

        gamma, ksi = compute_snr(sig2, noise_mu2, prev_ksi, snr_parameters)

        spp_parameters['ksi'] = ksi
        spp_parameters['gamma'] = gamma
        #pSpeech = spp(qk, spp_parameters)

        # 3 precise noise estimation
        #noise_parameters['P_speech'] = pSpeech
        noise_parameters['ksi'] = ksi
        noise_parameters['gamma'] = gamma
        noise_mu2 = noise_estimation(sig2, noise_mu2, noise_parameters, noise_method)

        gamma, ksi = compute_snr(sig2, noise_mu2, ksi, snr_parameters)

        gain_parameters['ksi'] = ksi
        gain_parameters['gamma'] = gamma
        hw = ns_gain(gain_parameters)

        xi_w = np.fft.ifft(hw * spec, nFFT, axis=0)
        xi_w = np.real(xi_w)
        xfinal[k:k + len2] = x_old + xi_w[0:len1]

        x_old = xi_w[len1:Slen]
        prev_ksi = ksi


    sf.write(output_file, xfinal, Srate)
    plt.figure(2)
    plt.specgram(xfinal, NFFT=Slen, Fs=Srate, noverlap=len2, cmap='jet')
    plt.ylim((0, 4000))
    plt.show()


if __name__=="__main__":
    main()


