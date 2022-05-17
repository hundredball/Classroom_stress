#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 10 16:44 2021

@author: hundredball
"""
import itertools

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
from scipy.integrate import simps

import config as cfg
import dataloader as dl

def get_bandpower(data, low = [4,7,13], high=[7,13,30], dB_scale = False):
    '''
    Calculate bandpower of theta, alpha, beta

    Parameters
    ----------
    data : list of 2d array 
        Time signal data
        i : example
        j : channel
        k : sample
    low : list
        lower bounds of band power
    high : list
        higher bounds of band power
    dB_scale : bool
        tranform band power into dB

    Returns
    -------
    powers : list of 2d array
        i : example
        j : channel
        k : band

    '''
    assert hasattr(data, '__iter__') and all((sample.ndim==2 for sample in data))
    assert hasattr(low, '__iter__') and hasattr(high, '__iter__')
    assert all((high[i]>=low[i]) for i in range(len(high)))
    assert isinstance(dB_scale, bool)
    
    print('Calculating the bandpower of time-series data...')

    powers = []
    psds = []
    for i, sample in enumerate(data):
        freqs, psd = signal.welch(sample, cfg.fs, nperseg=cfg.win, noverlap=cfg.win//2)
        
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]  # = 1/0.5 = 2
        
        # Find intersecting values in frequency vector
        idx = np.logical_and(freqs[:,np.newaxis] >= low, freqs[:,np.newaxis] <= high)
        idx = idx.T   # (65,3)->(3,65)
        
        # Compute the absolute power by approximating the area under the curve
        sample_powers = np.zeros((sample.shape[0],len(high)))
        for i_band in range(len(high)):
            idx_power = idx[i_band,:]
            sample_powers[:,i_band] = simps(psd[:,idx_power], dx=freq_res)
        
        # Transform into dB
        if dB_scale:
            sample_powers = 10*np.log10(sample_powers)
            psd = 10*np.log10(psd)
        
        powers.append(sample_powers)
        psds.append(psd)
        
    print('freqs: ', freqs)
        
    return powers, psds, freqs

if __name__ == '__main__':
    
    fs = 1000
    t = np.linspace(0,60,60*fs)
    f1, f2 = 20, 40
    time_signal = 0
    for i in range(1,51):
        time_signal += 1/i*np.cos(2*np.pi*i*t)
    time_signal = [time_signal[np.newaxis,:]]

    low, high = list(range(1,50)), list(range(2,51))
    powers, psds, freqs = get_bandpower(time_signal, fs, low, high)
    
    #freqs = range(1,50)
    indices = np.where(freqs<50)[0]
    psds = [psds[i][:,indices] for i in range(len(psds))]
    freqs = freqs[indices]
    plt.plot(freqs,psds[0][0])
    plt.xlabel('Hz')
    plt.ylabel('PSD')
    plt.savefig('./results/test_bandpower.png')
    