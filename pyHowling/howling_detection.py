#!/usr/bin/python

"""
Functions for howling frequency detection.

"""
from __future__ import division

import numpy as np

def papr(frame, threshold):
    """Peak-to-Avarage Power Ratio (PAPR)
    Returns all frequency indices where power is greater than avarage power + threshold,
    which are possible candidates where howling occurs.

    Args:
        frame: Spectrum of one frame.
        threshold: Power threshold value in dB.
         The returned candidates should have power greater than average power + threshold.

    Returns:
        A list of selected frequncy indices.
        A list of PAPR value for every freqency bin of this frame.
    """
    power = np.abs(frame)**2
    papr = np.zeros(len(power))
    avarage = np.mean(power)
    ret = []
    for i in range(len(frame)):
        papr[i] = 10*np.log10(power[i]/avarage) 
        if (papr[i] > threshold): 
            ret.append(i)
    return ret, papr

def ptpr(frame, threshold):
    """Peak-to-Threshold Power Ratio (PTPR)
    Returns all frequency indices where power is greater than threshold,
    which are possible candidates where howling occurs.

    Args:
        frame: Spectrum of one frame.
        threshold: Power threshold value in dB.

    Returns:
        A list of selected frequncy indices.
    """
    power =  np.abs(frame)**2
    ret = []
    for i in range(len(frame)):
        if (10*np.log10(power[i]) > threshold): 
            ret.append(i)
    return ret

#def phpr(frame, threshold):
"""Peak-to-Harmonic Power Ratio (PHPR)
       To-Do
    Args:

    Returns:

"""

def pnpr(frame, threshold):
    """Peak-to-Neighboring Power Ratio (PNPR)
    Returns all frequency indices of power peaks,
    which are greater than neighboring frequency bins by a threshold.

    Args:
        frame: Spectrum of one frame.
        threshold: Power threshold value in dB.

    Returns:
        A list of selected frequncy indices.
    """
    power =  frame**2
    ret = []
    for i in range(5, len(frame)-5):
        if (10*np.log10(power[i]/power[i-4]) > threshold 
        and 10*np.log10(power[i]/power[i-5]) > threshold 
        and 10*np.log10(power[i]/power[i+4]) > threshold 
        and 10*np.log10(power[i]/power[i+5]) > threshold ):
            ret.append(i)
    return ret

def ipmp(candidates, index):
    """Inerframe Peak Magnitude Persistence (IPMP)
    Temporal howling detection criteria. Candidate should meet criteria 
    in more than 3 frames out of 5 continuous frames.

    Args:
        candidates: nFreqs X nFrames
                    candidates[f][t] = 1 means a candidate at frequency[f]  at frame[t]
        index: Current frame index.

    Returns:
        A list of selected frequncy indices.
    """
    accu = np.zeros(candidates.shape[0])

    if(index == 2):
        accu = np.sum(candidates[:, :index+1], axis=1)
    elif (index == 3):
        accu = np.sum(candidates[:, :index+1], axis=1)
    else:
        accu = np.sum(candidates[:, index-4:index+1], axis=1)

    #ipmp = np.squeeze(np.argwhere(accu >=3))
    ipmp = [idx for idx, val in enumerate(accu) if val >= 3] 
    return ipmp


#def imsd(frame, threshold):

def screening(frame, candidates):
    """
    Screen the candidates. Only one frequency in neighboring several frequencies is needed

    Args:
        frame: Current frame spectrum.
        candidates: nFreqs X nFrames
                    candidates[f][t] = 1 means a candidate at frequency[f]  at frame[t]

    Returns:
        A list of selected frequncy indices.
    """
    ret = []
    for c in candidates:
        if len(ret)==0:
            ret.append(c)
        elif ret[len(ret)-1] > c-3 :
            if abs(frame[ret[len(ret)-1]]) < abs(frame[c]):
                ret[len(ret)-1] = c
        else:
            ret.append(c)
    return ret
