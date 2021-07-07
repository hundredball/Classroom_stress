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
from scipy import signal
from scipy import interpolate
from scipy.integrate import simps

import dataloader as dl

def get_bandpower(data, fs, low = [4,7,13], high=[7,13,30], dB_scale = False):
    '''
    Calculate bandpower of theta, alpha, beta

    Parameters
    ----------
    data : list of 2d array 
        Time signal data
        i : example
        j : channel
        k : sample
    fs : int
        sampling rate
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
    assert isinstance(fs, int) and fs>0 
    assert hasattr(low, '__iter__') and hasattr(high, '__iter__')
    assert all((high[i]>=low[i]) for i in range(len(high)))
    assert isinstance(dB_scale, bool)
    
    print('Calculating the bandpower of time-series data...')
    # Define window length
    win = 5*fs
    
    powers = []
    for i, sample in enumerate(data):
        freqs, psd = signal.welch(sample, fs, nperseg=win, noverlap=win/2)
        
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
        
        powers.append(sample_powers)
        
    print('freqs: ', freqs)
        
    return powers

if __name__ == '__main__':
    
    fs = 500
    t = np.linspace(0,60,60*fs)
    f1, f2 = 20, 40
    signal = 0
    for i in range(1,51):
        signal += np.random.rand()*np.cos(2*np.pi*i*t)
    signal = [signal[np.newaxis,:]]

    low, high = list(range(1,50)), list(range(2,51))
    powers = bandpower.get_bandpower(signal, fs, low, high)
    
    freqs = range(1,50)
    plt.plot(freqs,powers[0][0])
    