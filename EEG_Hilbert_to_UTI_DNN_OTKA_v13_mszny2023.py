'''
Written by Tamas Gabor Csapo <csapot@tmit.bme.hu>
First version Sep 15, 2021
- WaveGlow sample from Csaba Zainko
- restructured for EEG-to-UTI on Nov 14, 2022
- simple FC-DNN
- keras implementation of:
Csapó Tamás Gábor, Arthur Frigyes Viktor, Nagy Péter, Boncz Ádám, ,,A beszéd artikulációs mozgásának predikciója agyi jel alapján - kezdeti eredmények'', XIX. Magyar Számítógépes Nyelvészeti Konferencia (MSZNY2023), Szeged,
http://smartlab.tmit.bme.hu/downloads/pdf/csapot/Csapo-et-al-mszny2023-cikk.pdf

# TODO: clean the code...
'''

import WaveGlow_functions
# from memory_profiler import profile
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import torch
import datetime
import gc
import os
import pickle
import sys
import argparse
from datetime import datetime, timedelta

from detect_peaks import detect_peaks
from subprocess import call, check_output, run

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy
import scipy.signal
import scipy.stats
import scipy.io.wavfile
import scipy.fftpack
import scipy.io as sio
import skimage.transform
import soundfile as sf
import tensorflow

# '''
# do not use all GPU memory
from keras.backend.tensorflow_backend import set_session
config = tensorflow.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True 
set_session(tensorflow.Session(config=config))
# '''

import scipy
import scipy.io
# import scipy.io.wavfile
import scipy.io.wavfile as io_wav

# EEG signal processing / Hilbert transform,
# from https://github.com/neuralinterfacinglab/SingleWordProductionDutch

#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def extractHG(data, sr, windowLength=0.05, frameshift=0.01, bandpass_min=70, bandpass_max=170):
    """
    Window data and extract frequency-band envelope using the hilbert transform
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    #Number of windows
    numWindows = int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    #Filter High-Gamma Band
    # sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    sos = scipy.signal.iirfilter(4, [bandpass_min/(sr/2),bandpass_max/(sr/2)],btype='bandpass',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate first harmonic of line noise
    # sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    # data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate second harmonic of line noise
    # sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    # data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Create feature space
    data = np.abs(hilbert3(data))
    feat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:] = np.mean(data[start:stop,:],axis=0)
    return feat

def stackFeatures(features, modelOrder=4, stepSize=5):
    """
    Add temporal context to each window by stacking neighboring feature vectors
    
    Parameters
    ----------
    features: array (windows, channels)
        Feature time series
    modelOrder: int
        Number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate for frameshift)
    Returns
    ----------
    featStacked: array (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    featStacked=np.zeros((features.shape[0]-(2*modelOrder*stepSize),(2*modelOrder+1)*features.shape[1]))
    for fNum,i in enumerate(range(modelOrder*stepSize,features.shape[0]-modelOrder*stepSize)):
        ef=features[i-modelOrder*stepSize:i+modelOrder*stepSize+1:stepSize,:]
        featStacked[fNum,:]=ef.flatten() #Add 'F' if stacked the same as matlab
    return featStacked

# read_ult reads in *.ult file from AAA
def read_ult(filename, NumVectors, PixPerVector):
    # read binary file
    ult_data = np.fromfile(filename, dtype='uint8')
    ult_data = np.reshape(ult_data, (-1, NumVectors, PixPerVector))
    return ult_data

def read_wav(filename):
    (Fs, x) = io_wav.read(filename)
    return (x, Fs)

def write_wav(x, Fs, filename):
    # scaled = np.int16(x / np.max(np.abs(x)) * 32767)
    io_wav.write(filename, Fs, np.int16(x))

def find_sync_from_ultrasound(ult_sync):
    sync_data = ult_sync.copy()
    
    sync_threshold = np.max(sync_data) * 0.6
    # clip
    for s in range(len(sync_data)):
        if sync_data[s] > sync_threshold:
            sync_data[s] = sync_threshold

    # find peeks in the derivative
    # peakind1 = detect_peaks(np.gradient(sync_data), mph=1000, mpd=10, threshold=0, edge='both')
    peakind1 = detect_peaks(sync_data, mph=0.9*sync_threshold, mpd=10, threshold=0, edge='rising')
    
    '''
    plt.figure(figsize=(18,4))
    plt.plot(sync_data)
    plt.plot(np.gradient(sync_data), 'r')
    for i in range(len(peakind1)):
        plt.plot(peakind1[i], sync_data[peakind1[i]], 'gx')
        # plt.plot(peakind2[i], sync_data[peakind2[i]], 'r*')
    # plt.xlim(2000, 6000)
    plt.show()
    '''
    
    return list(peakind1)

def find_sync_from_eeg_aux(eeg_aux):
    
    sync_data = -eeg_aux.copy()
    sync_data -= np.mean(sync_data[0:1000])

    # clip
    sync_threshold = np.max(sync_data) * 0.5
    for s in range(len(sync_data)):
        if sync_data[s] > sync_threshold:
            sync_data[s] = sync_threshold
    
    # find peeks
    peakind1 = detect_peaks(sync_data, mph=0.9*sync_threshold, mpd=8, threshold=0, edge='rising')
    
    
    # there is at least 50ms silence before the actual pulses!
    for i in range(len(peakind1) - 1):
        if (peakind1[i] - peakind1[i-1]) > 50:
            # print('first peak after silence: ', peakind1[i])
            peakind1 = peakind1[i:]
            break
            
    
    '''
    # figure for debugging
    plt.figure(figsize=(18,4))
    plt.plot(sync_data)
    plt.plot(np.gradient(sync_data), 'r')
    for i in range(len(peakind1)):
        plt.plot(peakind1[i], sync_data[peakind1[i]], 'gx')
        # plt.plot(peakind2[i], sync_data[peakind2[i]], 'r*')
    # plt.xlim(37000, 39000)
    plt.show()    
    '''
    
    return list(peakind1)

def cut_and_resample_wav(filename_wav_in, init_offset, hop_length_UTI, Fs_target):
    filename_no_ext = filename_wav_in.replace('.wav', '')
    
    filename_param = filename_no_ext + '.param'
    filename_wav_out = filename_no_ext + '_cut_22k.wav'
    
    # resample speech using SoX
    command = 'sox ' + filename_wav_in + ' -r ' + str(Fs_target) + ' ' + \
              filename_no_ext + '_22k.wav'
    call(command, shell=True)
    
    # volume normalization using SoX
    command = 'sox --norm=-3 ' + filename_no_ext + '_22k.wav' + ' ' + \
              filename_no_ext + '_22k_volnorm.wav'
    call(command, shell=True)
    
    # cut from wav the signal the part where there are ultrasound frames
    (speech_wav_data, Fs_wav) = read_wav(filename_no_ext + '_22k_volnorm.wav')
    # speech_wav_data = speech_wav_data[init_offset - hop_length_UTI : ]
    speech_wav_data = speech_wav_data[init_offset - hop_length_UTI : ]
    write_wav(speech_wav_data, Fs_wav, filename_wav_out)
    
    # remove temp files
    os.remove(filename_no_ext + '_22k.wav')
    os.remove(filename_no_ext + '_22k_volnorm.wav')
    
    print(filename_no_ext + ' - resampled, volume normalized, and cut to start with ultrasound')


# sample from Csaba
def main(datasize, delay):
    
    print('started')
    
    # delay = 1123
    
    # initial delay:
    # 0 ms, 100 ms, 200 ms, ... 1000 ms -> 10
    # 1s...10s -> 10x10 = 100 trainings
    
        
    # WaveGlow / Tacotron2 / STFT parameters for audio data
    # samplingFrequency = 16000
    samplingFrequency = 22050
    samplingFrequency_EEG = 1000
    winL_EEG = 0.05
    # frameshift_EEG = 0.01 # 10 ms
    frameshift_EEG = 0.012 # 12 ms
    # modelOrder_EEG = 1
    # modelOrder_EEG = 2
    modelOrder_EEG = 4
    # modelOrder_EEG = 10
    stepSize_EEG = 5
    n_melspec = 80
    hop_length = 220  # 10 ms
    hop_length_UTI = 270  # 12 ms at 22 kHz
    NumVectors = 64
    PixPerVector = 842
    PixPerVector_resized = 128
    
    stft = WaveGlow_functions.TacotronSTFT(
        filter_length=1024,
        hop_length=hop_length_UTI,
        win_length=1024,
        n_mel_channels=n_melspec,
        sampling_rate=samplingFrequency,
        mel_fmin=0,
        mel_fmax=8000)
    
    dir_data = '/shared/TTK-EEG-UTI/spkr01_ses01_PPBA/'
    # dir_data = '/shared/TTK-EEG-UTI/spkr01_ses02_PPBA/'
    # dir_data = '/shared/TTK-EEG-UTI/spkr01_ses03_nonsense_words/'
    file_vhdr = '/shared/TTK-EEG-UTI/01_experiment_2022_10_06/eeg/Tamas_Viktor_01.vhdr'
    # file_vhdr = '/shared/TTK-EEG-UTI/01_experiment_2022_10_06/eeg/Tamas_Viktor_02.vhdr'
    # file_vhdr = '/shared/TTK-EEG-UTI/01_experiment_2022_10_06/eeg/Tamas_Viktor_03.vhdr'
    
    # Hungarian data from TTK
    raw_eeg = mne.io.read_raw_brainvision(file_vhdr, preload='True')
    
    
    
    print('EEG loaded')

    print(raw_eeg.info)
    # print(raw_eeg.info['meas_date'])
    # raise
    
    
    # raw_eeg.plot()
    # plt.show()
    ch_sync = raw_eeg._data[63]
    
    '''
    temp_sync = raw_eeg.copy()._data[63] - np.mean(raw_eeg._data[63])
    temp_sync = np.int16(temp_sync / np.max(np.abs(temp_sync)) * 32767)
    write_wav(temp_sync, 1000, 'temp_sync.wav')
    '''
    
    # plt.plot(ch_sync[int(37.5*1000):int(37.8*1000)])
    # plt.show()
    # raise
    # EEG AUX 1st sync pulse: 37.5*1000+65 = 37565
    # EEG AUX 2nd sync pulse: 37.5*1000+81 = 37581 # 16ms
    # EEG AUX 3rd sync pulse: 37.5*1000+81 = 37593 # 12ms
    # print('start find sync from EEG AUX')
    # eeg_aux_sync = find_sync_from_eeg_aux(ch_sync)
    # print('finished find sync from EEG AUX, first peak at (1000 Hz) : ', eeg_aux_sync[0])
    
    # EEG start time:                    2022-10-06 13:43:52 UTC
    # time delay in EEG:                 ~ +8mins
    # first speech time: (in EEG AUX)    +37.565s  = 13:44:29.565
    # first speech time: (in UTI)        2022. 10. 06. 13:36:30
    # first speech time: (UTI to EEG     2022. 10. 06. 13:36:30   + 7min 59sec
    # 13:36:30   -> 37 sec, i.e. EEG started in UTI time at: 13:35:53
    
    n_max_frames = datasize * 600
    # lookback_eeg = 10
    lookback_eeg = 5
    n_freq_bands = 4
    # n_mel_channels_eeg_low = 8
    # window_size_eeg = 100  # 100 ms
    # n_eeg_channels = 64
    n_eeg_channels = 71
    n_fft = 17
    eeg_fft = np.empty((n_max_frames, n_freq_bands, n_eeg_channels * (2 * modelOrder_EEG + 1) ))
    melspec = np.empty((n_max_frames, n_melspec))
    ult = np.empty((n_max_frames, NumVectors, PixPerVector_resized))
    eegmel_size = 0

    # raw_eeg = raw_eeg.crop(tmin=delay/1000, tmax=datasize)
    
    # resample EEG signal to 100 Hz
    # raw_eeg = raw_eeg.resample(sfreq=100)
    
    
    # collect all speech wav files
    wav_files_all = []
    
    # dir_data = '/shared/TTK-EEG-UTI/spkr01_ses03_nonsense_words/'
    # dir_data = 'c:/Users/csapot/Downloads/brain2speech/2022.10.06_TTK-EEG-UTI/session1_PPBA/'
    if os.path.isdir(dir_data):
        for file in sorted(os.listdir(dir_data)):
            if file.endswith('_speech.wav'): # and '069' in file:
                if not os.path.isfile(dir_data + file.replace('_speech.wav', '_speech_cut_22k.wav')):
                    # resample and cut according to psync
                    print('resample and cut according to psync', os.path.basename(file))
                    (Fs, sync_data) = io_wav.read(dir_data + file.replace('_speech.wav', '_sync.wav'))
                    ult_sync_pulses = find_sync_from_ultrasound(sync_data)
                    # 44 kHz to 22 kHz
                    init_offset = int(ult_sync_pulses[0] / 2)
                    cut_and_resample_wav(dir_data + file, init_offset, hop_length_UTI, samplingFrequency)
                    # raise
                wav_files_all += [dir_data + file.replace('_speech.wav', '')]
                
    
    # wav_files_all = wav_files_all[0:2]
    # wav_files_all = wav_files_all[0:24]
    # print(wav_files_all)
    
    # date_EEG_start = datetime.strptime('2022-10-06 13:43:52', '%Y-%m-%d %H:%M:%S') # 01
    #2022-10-06 13:43:52.922777+00:00
    # print()
    date_EEG_start = raw_eeg.info['meas_date'].replace(tzinfo=None)
    print('date_EEG_start', date_EEG_start)
    
    # date_EEG_start = datetime.strptime('2022-10-06 14:46:30', '%Y-%m-%d %H:%M:%S') # 03
    date_EEG_start_in_speech_time = date_EEG_start - timedelta(minutes=7, seconds=58.5) # 01
    # date_EEG_start_in_speech_time = date_EEG_start - timedelta(minutes=7, seconds=59.5) # 02
    # date_EEG_start_in_speech_time = date_EEG_start - timedelta(minutes=8, seconds=2.5) # 03/1-25
    # date_EEG_start_in_speech_time = date_EEG_start - timedelta(minutes=8, seconds=1.5) # 03/26
    # print(date_EEG_start_in_speech_time)
    # raise
    
    for file in wav_files_all:
        # read txt file
        with open(file + '.txt', 'rt', encoding='latin2',) as txt_file:
            print(txt_file)
            txt_content = txt_file.readlines()
            # 2022. 10. 06. 13:36:30
        txt_date = txt_content[1][:-1]
        speech_start = datetime.strptime(txt_date, '%Y. %m. %d. %H:%M:%S')
        speech_start_from_EEG_beginning = (speech_start - date_EEG_start_in_speech_time).total_seconds()
        
        (Fs, sync_data) = io_wav.read(file + '_sync.wav')
        # ult_sync_pulses = find_sync_from_ultrasound(sync_data) # at 44 kHz
        speech_length = len(sync_data) / Fs
        speech_end_from_EEG_beginning = speech_start_from_EEG_beginning + speech_length
        print(date_EEG_start_in_speech_time, speech_start, speech_start_from_EEG_beginning, speech_end_from_EEG_beginning)
        
        # current_raw_eeg = raw_eeg.crop(tmin=speech_start_from_EEG_beginning, tmax=speech_end_from_EEG_beginning)
        
        # current_sync_from_EEG = current_raw_eeg._data[63]
        
        current_sync_from_EEG = ch_sync[int(speech_start_from_EEG_beginning * samplingFrequency_EEG) : int(speech_end_from_EEG_beginning * samplingFrequency_EEG)]
        
        current_pulses_from_EEG = find_sync_from_eeg_aux(current_sync_from_EEG)
        print('pulses: ', current_pulses_from_EEG[0:5])
        
        speech_start_from_EEG_beginning_correct = speech_start_from_EEG_beginning + current_pulses_from_EEG[0] / samplingFrequency_EEG
        speech_end_from_EEG_beginning_correct = speech_start_from_EEG_beginning_correct + speech_length
        
        print('speech start: ', speech_start_from_EEG_beginning_correct)
        
        current_raw_eeg = raw_eeg.copy().crop(tmin=speech_start_from_EEG_beginning_correct, tmax=speech_end_from_EEG_beginning_correct)
        
        # resample EEG from 1kHz to 100 Hz
        # current_raw_eeg_resampled = current_raw_eeg.copy().resample(sfreq=100)
        
        n_frames_eeg = int(len(current_raw_eeg) / 1) # 10ms at 100Hz
        
        wavfile = file + '_speech_cut_22k.wav'
        mel_data = WaveGlow_functions.get_mel(wavfile, stft)
        mel_data = np.fliplr(np.rot90(mel_data.data.numpy(), axes=(1, 0)))
        
        # temp:plot wav and sync
        
        ult_data = read_ult(file + '.ult', NumVectors, PixPerVector)
        
        # '''
        temp_sync = current_raw_eeg._data[63] - np.mean(current_raw_eeg._data[63])
        temp_sync = np.int16(temp_sync / np.max(np.abs(temp_sync)) * 32767)
        # write_wav(temp_sync, 1000, 'temp_sync.wav')
        (Fs, wav_data) = io_wav.read(wavfile)
        plt.figure(figsize=(18,8))
        plt.subplot(211)
        plt.plot(wav_data)
        plt.subplot(212)
        plt.plot(temp_sync)
        plt.savefig(file + '_sync.png')
        # plt.show()
        plt.close()
        # '''
        
        # raise
        
        n_frames_mel = len(mel_data)
        
        print('mel_data', mel_data.shape)
        
        current_raw_eeg_data = current_raw_eeg._data.copy()            
        
        # zero out sync channel
        current_raw_eeg_data[63] = 0
        
        #Extract HG features
        print('calculating Hilbert...', current_raw_eeg_data.shape)
        # eeg_fft = np.empty((n_max_frames, n_freq_bands, n_eeg_channels * (2 * modelOrder_EEG + 1) ))
        feat_Hilbert_1 = extractHG(np.rot90(current_raw_eeg_data),samplingFrequency_EEG, windowLength=winL_EEG,frameshift=frameshift_EEG, bandpass_min=1, bandpass_max=50)
        feat_Hilbert_2 = extractHG(np.rot90(current_raw_eeg_data),samplingFrequency_EEG, windowLength=winL_EEG,frameshift=frameshift_EEG, bandpass_min=51, bandpass_max=100)
        feat_Hilbert_3 = extractHG(np.rot90(current_raw_eeg_data),samplingFrequency_EEG, windowLength=winL_EEG,frameshift=frameshift_EEG, bandpass_min=101, bandpass_max=150)
        feat_Hilbert_4 = extractHG(np.rot90(current_raw_eeg_data),samplingFrequency_EEG, windowLength=winL_EEG,frameshift=frameshift_EEG, bandpass_min=151, bandpass_max=200)
        
        '''
        plt.subplot(411)
        plt.imshow(np.rot90(feat_Hilbert_1[0:500]))
        plt.gray()
        plt.subplot(412)
        plt.imshow(np.rot90(feat_Hilbert_2[0:500]))
        plt.gray()
        plt.subplot(413)
        plt.imshow(np.rot90(feat_Hilbert_3[0:500]))
        plt.gray()
        plt.subplot(414)
        plt.imshow(np.rot90(feat_Hilbert_4[0:500]))
        plt.gray()
        plt.show()
        '''
        # raise
        
        print('feat_Hilbert', feat_Hilbert_1.shape)
        
        #Stack features
        feat_Hilbert_1 = stackFeatures(feat_Hilbert_1,modelOrder=modelOrder_EEG,stepSize=stepSize_EEG)
        feat_Hilbert_2 = stackFeatures(feat_Hilbert_2,modelOrder=modelOrder_EEG,stepSize=stepSize_EEG)
        feat_Hilbert_3 = stackFeatures(feat_Hilbert_3,modelOrder=modelOrder_EEG,stepSize=stepSize_EEG)
        feat_Hilbert_4 = stackFeatures(feat_Hilbert_4,modelOrder=modelOrder_EEG,stepSize=stepSize_EEG)
        
        
        print('feat_Hilbert/stacked', feat_Hilbert_1.shape)
        print('mel_data', mel_data.shape)
        mel_data = mel_data[modelOrder_EEG * stepSize_EEG : - modelOrder_EEG * stepSize_EEG]
        print('mel_data/stacked', mel_data.shape)
        # raise
        
        
        ult_data = read_ult(file + '.ult', NumVectors, PixPerVector)
        
        n_frames_eeg = len(feat_Hilbert_1)
        n_frames_mel = len(mel_data)
        n_frames = np.min((n_frames_eeg, n_frames_mel))
        print(n_frames_eeg, n_frames_mel, n_frames)
        feat_Hilbert_1 = feat_Hilbert_1[0:n_frames]
        feat_Hilbert_2 = feat_Hilbert_2[0:n_frames]
        feat_Hilbert_3 = feat_Hilbert_3[0:n_frames]
        feat_Hilbert_4 = feat_Hilbert_4[0:n_frames]
        mel_data = mel_data[0:n_frames]
        
        
        # raise
        
        # for n in range(n_frames - lookback_eeg):
        print(eegmel_size, n_frames, eegmel_size + n_frames, eeg_fft[eegmel_size : eegmel_size + n_frames, 0, :].shape, feat_Hilbert_1.shape)
        eeg_fft[eegmel_size : eegmel_size + n_frames, 0, :] = feat_Hilbert_1
        eeg_fft[eegmel_size : eegmel_size + n_frames, 1, :] = feat_Hilbert_2
        eeg_fft[eegmel_size : eegmel_size + n_frames, 2, :] = feat_Hilbert_3
        eeg_fft[eegmel_size : eegmel_size + n_frames, 3, :] = feat_Hilbert_4
        # raise
        
        melspec[eegmel_size : eegmel_size + n_frames] = mel_data
        
        print('resizing ultrasound...')
        for i in range(n_frames):
            ult[eegmel_size + i] = skimage.transform.resize(ult_data[i], (NumVectors, PixPerVector_resized), preserve_range=True) / 255
        
        eegmel_size += n_frames
        
        '''
        plt.subplot(211)
        plt.imshow(np.rot90(melspec[0 : eegmel_size]))
        plt.gray()
        plt.subplot(212)
        plt.imshow(np.rot90(ult[0 : eegmel_size, 32, 47:128].squeeze()))
        plt.gray()
        plt.show()
        '''
        
        '''
        print('calculating FFT for ', os.path.basename(file))                
        for e in range(n_eeg_channels):
            f, t, eeg_spectrogram = scipy.signal.spectrogram(current_raw_eeg_resampled._data[e], fs = 100, nperseg=5, noverlap=4, nfft = (n_fft-1)*2)
            
            print('feat_Hilbert', feat_Hilbert.shape)
            print('eeg_spectrogram', eeg_spectrogram.shape)
            
            raise
            
            for n in range(n_frames - lookback_eeg):
                
                for m in range(n_fft):
                    eeg_fft[eegmel_size + n, e, m] = eeg_spectrogram[m, n: n + lookback_eeg]
        '''
        
        # for n in range(n_frames - lookback_eeg):
            # melspec[eegmel_size + n] = mel_data[n + lookback_eeg]

        # eegmel_size += n

            
    # get roughly the location of speech in EEG
    # EEG start time:                    2022-10-06 13:43:52 UTC
    # time delay in EEG:                 ~ +8mins
    # first speech time: (in EEG AUX)    +37.565s  = 13:44:29.565
    # first speech time: (in UTI)        2022. 10. 06. 13:36:30
    # first speech time: (UTI to EEG     2022. 10. 06. 13:36:30   + 7min 59sec
    # 13:36:30   -> 37 sec, i.e. EEG started in UTI time at: 13:35:53
    
    # raise
    
    
    
    
    
    print('n_frames_all: ', eegmel_size)

    ult = ult.reshape(-1, NumVectors * PixPerVector_resized)

    eeg_fft = eeg_fft[0: eegmel_size]
    melspec = melspec[0: eegmel_size]
    ult = ult[0: eegmel_size]

    # plt.imshow(np.rot90(melspec))    
    # plt.gray()
    # plt.show()
    # raise

    
    print(eeg_fft.shape)
    print(melspec.shape)
    print(ult.shape)
    
    '''
    plt.subplot(211)
    plt.imshow(np.rot90(melspec[-n_frames : ]))
    plt.gray()
    plt.subplot(212)
    plt.imshow(np.rot90(ult[-n_frames : , 32, 47:128].squeeze()))
    plt.gray()
    plt.show()
    '''

    # raise

    # scale EEG data to [0-1]
    # eeg = eeg_fft.reshape(-1, n_eeg_channels * n_fft * lookback_eeg)
    eeg = eeg_fft.reshape(-1, n_freq_bands * n_eeg_channels * (2 * modelOrder_EEG + 1))
    # eeg = np.empty((n_max_frames, n_freq_bands, n_eeg_channels * (2 * modelOrder_EEG + 1) ))
    # eeg = eeg_fft
    
    # split to train-validation-test: 80-10-10%
    eeg_train = eeg[0 : int(len(eeg) * 0.8)]
    eeg_valid = eeg[int(len(eeg) * 0.8) : int(len(eeg) * 0.9)]
    eeg_test =  eeg[int(len(eeg) * 0.9) : ]    
    
    melspec_train = melspec[0 : int(len(eeg) * 0.8)]
    melspec_valid = melspec[int(len(eeg) * 0.8) : int(len(eeg) * 0.9)]
    melspec_test =  melspec[int(len(eeg) * 0.9) : ]
    
    ult_train = ult[0 : int(len(eeg) * 0.8)]
    ult_valid = ult[int(len(eeg) * 0.8) : int(len(eeg) * 0.9)]
    ult_test =  ult[int(len(eeg) * 0.9) : ]
    
    
    # scale input to [0-1]
    eeg_scaler = MinMaxScaler()
    # eeg_scaler = StandardScaler(with_mean=True, with_std=True)
    eeg_train_scaled = eeg_scaler.fit_transform(eeg_train)
    eeg_valid_scaled = eeg_scaler.transform(eeg_valid)
    eeg_test_scaled  = eeg_scaler.transform(eeg_test)
    
    del raw_eeg
    # del eeg

    # scale outpit mel-spectrogram data to zero mean, unit variances
    melspec_scaler = StandardScaler(with_mean=True, with_std=True)
    melspec_train_scaled = melspec_scaler.fit_transform(melspec_train)
    melspec_valid_scaled = melspec_scaler.transform(melspec_valid)
    melspec_test_scaled  = melspec_scaler.transform(melspec_test)



    # 5 hidden layers, with 1000 neuron on each layer
    model = Sequential()
    model.add(
        Dense(
            1000,
            input_dim=n_freq_bands * n_eeg_channels * (2 * modelOrder_EEG + 1),
            kernel_initializer='normal',
            activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(
        Dense(
            NumVectors * PixPerVector_resized,
            kernel_initializer='normal',
            activation='linear'))

    # compile model
    model.compile(
        loss='mean_squared_error',
        metrics=['mean_squared_error'],
        optimizer='adam')
    earlystopper = EarlyStopping(
        monitor='val_mean_squared_error',
        min_delta=0.0001,
        patience=3,
        verbose=1,
        mode='auto')

    print(model.summary())

    if not (os.path.isdir('models_EEG_to_UTI/')):
        os.mkdir('models_EEG_to_UTI/')

    # early stopping to avoid over-training
    # csv logger
    current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format(
        date=datetime.now())
    print(current_date)
    # n_eeg_channels * (2 * modelOrder_EEG + 1)
    model_name = 'models_EEG_to_UTI/EEG-Hilbert_to_UTI_DNN_modelOrder-' + str(modelOrder_EEG).zfill(2) + '_freqBands-4_' + current_date
    logger = CSVLogger(model_name + '.csv', append=True, separator=';')
    checkp = ModelCheckpoint(
        model_name +
        '_weights_best.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')

    # save model
    model_json = model.to_json()
    with open(model_name + '_model.json', "w") as json_file:
        json_file.write(model_json)

    # serialize scalers to pickle
    # pickle.dump(eeg_scaler, open(model_name + '_eeg_scaler.sav', 'wb'))
    # pickle.dump(melspec_scaler, open(model_name + '_melspec_scaler.sav', 'wb'))

    # Run training
    history = model.fit(eeg_train_scaled, ult_train,
                        epochs=100, batch_size=128, shuffle=True, verbose=1,
                        callbacks=[earlystopper, logger, checkp],
                        validation_data=(eeg_valid_scaled, ult_valid),
                        )

    # here the training of the DNN is finished
    # load back best weights
    model.load_weights(model_name + '_weights_best.h5')
    # remove model file
    os.remove(model_name + '_weights_best.h5')
           
    # melspec_predicted = model.predict(eeg_test_scaled[0:500])
    ult_predicted = model.predict(eeg_test_scaled[0:500])
    # melspec_predicted = melspec_predicted[0:500]
    # test_melspec = test_melspec[]
    
    ult_predicted = ult_predicted.reshape(-1, NumVectors, PixPerVector_resized)
    ult_test = ult_test.reshape(-1, NumVectors, PixPerVector_resized)
    
    np.save(model_name + '_TTK_ult_orig.npy', ult_test[0:500])
    
    # melspec_predicted_unscaled = melspec_scaler.inverse_transform(melspec_predicted)
    np.save(model_name + '_TTK_ult_predicted.npy', ult_predicted)
    
    # '''
    plt.subplot(211)
    # plt.imshow(eeg_train_scaled[100].reshape(n_eeg_channels,(2 * modelOrder_EEG + 1)))
    # n_freq_bands * n_eeg_channels * (2 * modelOrder_EEG + 1)
    plt.imshow(eeg_train_scaled[100].reshape(n_eeg_channels,n_freq_bands*(2 * modelOrder_EEG + 1)))
    plt.gray()
    plt.subplot(212)
    # plt.plot(melspec_train_scaled[100])
    plt.imshow(np.rot90(ult_train[100].reshape(NumVectors, PixPerVector_resized)))
    plt.gray()
    # plt.show()
    plt.savefig(model_name + '_EEG_scaled.png')
    plt.close()
    # '''
    
    plt.subplot(411)
    plt.imshow(np.rot90(melspec_test[0:500]))    
    plt.gray()
    plt.subplot(412)
    plt.imshow(np.rot90(melspec_test_scaled[0:500]))    
    plt.gray()
    plt.subplot(413)
    plt.imshow(np.rot90(ult_test[0 : 500, 32, 47:128].squeeze()))
    plt.gray()
    plt.subplot(414)
    # plt.imshow(np.rot90(melspec_predicted[0:500]))
    plt.imshow(np.rot90(ult_predicted[ : , 32, 47:128].squeeze()))
    plt.gray()
    # plt.subplot(414)
    # plt.imshow(np.rot90(melspec_predicted_unscaled[0:500]))
    plt.gray()
    plt.savefig(model_name + '_TTK_UTI_predicted.png')
    # plt.show()
    plt.close()
    # melspec = model.predict(eeg_scaled)
    # print(melspec)
    # print(melspec.shape)
    
    tensorflow.keras.backend.clear_session()
    
    
    # remove unnecessary variables
    # del raw_eeg
    # del eeg
    # del mel_data
    # del melspec
    # del eeg_scaled
    # del melspec_scaled
    
    gc.collect()
    
    


datasize = 50
delay = 0
main(datasize, delay)
    