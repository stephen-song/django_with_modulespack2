# -*- coding: utf-8 -*-
# !/usr/bin/env python

import os,time

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy import signal

from API.ModuleLib.UtilLib import params as hp

def load_wav(path):
  return librosa.core.load(path, sr=16000)[0]

def read_wav(path, sr, duration=None, mono=True):
    wav, _ = librosa.load(path, mono=mono, sr=sr, duration=duration)
    return wav


def write_wav(wav, sr, path, format='wav', subtype='PCM_16'):
    sf.write(path, wav, sr, format=format, subtype=subtype)


def read_mfcc(prefix):
    filename = '{}.mfcc.npy'.format(prefix)
    mfcc = np.load(filename)
    return mfcc


def write_mfcc(prefix, mfcc):
    filename = '{}.mfcc'.format(prefix)
    np.save(filename, mfcc)


def read_spectrogram(prefix):
    filename = '{}.spec.npy'.format(prefix)
    spec = np.load(filename)
    return spec


def write_spectrogram(prefix, spec):
    filename = '{}.spec'.format(prefix)
    np.save(filename, spec)


def split_wav(wav, top_db):
    intervals = librosa.effects.split(wav, top_db=top_db)
    wavs = map(lambda i: wav[i[0]: i[1]], intervals)
    return wavs


def trim_wav(wav):
    wav, _ = librosa.effects.trim(wav)
    return wav


def fix_length(wav, length):
    if len(wav) != length:
        wav = librosa.util.fix_length(wav, length)
    return wav


def crop_random_wav(wav, length):
    """
    Randomly cropped a part in a wav file.
    :param wav: a waveform
    :param length: length to be randomly cropped.
    :return: a randomly cropped part of wav.
    """
    assert (wav.ndim <= 2)
    assert (type(length) == int)

    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - length)), 1)[0]
    end = start + length
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def mp3_to_wav(src_path, tar_path):
    """
    Read mp3 file from source path, convert it to wav and write it to target path. 
    Necessary libraries: ffmpeg, libav.
    :param src_path: source mp3 file path
    :param tar_path: target wav file path
    """
    basepath, filename = os.path.split(src_path)
    os.chdir(basepath)
    AudioSegment.from_mp3(src_path).export(tar_path, format='wav')


def prepro_audio(source_path, target_path, format=None, sr=None, db=None):
    """
    Read a wav, change sample rate, format, and average decibel and write to target path.
    :param source_path: source wav file path
    :param target_path: target wav file path
    :param sr: sample rate.
    :param format: output audio format.
    :param db: decibel.
    """
    sound = AudioSegment.from_file(source_path, format)
    if sr:
        sound = sound.set_frame_rate(sr)
    if db:
        change_dBFS = db - sound.dBFS
        sound = sound.apply_gain(change_dBFS)
    sound.export(target_path, 'wav')


def _split_path(path):
    """
    Split path to basename, filename and extension. For example, 'a/b/c.wav' => ('a/b', 'c', 'wav')
    :param path: file path
    :return: basename, filename, and extension
    """
    basepath, filename = os.path.split(path)
    filename, extension = os.path.splitext(filename)
    return basepath, filename, extension


def wav2spec(wav, n_fft, win_length, hop_length, time_first=True):
    """
    Get magnitude and phase spectrogram from waveforms.
    Parameters
    ----------
    wav : np.ndarray [shape=(n,)]
        The real-valued waveform.
    n_fft : int > 0 [scalar]
        FFT window size.
    win_length  : int <= n_fft [scalar]
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.
    hop_length : int > 0 [scalar]
        Number audio of frames between STFT columns.
    time_first : boolean. optional.
        if True, time axis is followed by bin axis. In this case, shape of returns is (t, 1 + n_fft/2)
    Returns
    -------
    mag : np.ndarray [shape=(t, 1 + n_fft/2) or (1 + n_fft/2, t)]
        Magnitude spectrogram.
    phase : np.ndarray [shape=(t, 1 + n_fft/2) or (1 + n_fft/2, t)]
        Phase spectrogram.
    """
    stft = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(stft)
    phase = np.angle(stft)

    if time_first:
        mag = mag.T
        phase = phase.T

    return mag, phase


def spec2wav(mag, n_fft, win_length, hop_length, num_iters=30, phase=None):
    """
    Get a waveform from the magnitude spectrogram by Griffin-Lim Algorithm.
    Parameters
    ----------
    mag : np.ndarray [shape=(1 + n_fft/2, t)]
        Magnitude spectrogram.
    n_fft : int > 0 [scalar]
        FFT window size.
    win_length  : int <= n_fft [scalar]
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.
    hop_length : int > 0 [scalar]
        Number audio of frames between STFT columns.
    num_iters: int > 0 [scalar]
        Number of iterations of Griffin-Lim Algorithm.
    phase : np.ndarray [shape=(1 + n_fft/2, t)]
        Initial phase spectrogram.
    Returns
    -------
    wav : np.ndarray [shape=(n,)]
        The real-valued waveform.
    """
    assert (num_iters > 0)
    if phase is None:
        phase = np.pi * np.random.rand(*mag.shape)
    stft = mag * np.exp(1.j * phase)
    wav = None
    for i in range(num_iters):
        wav = librosa.istft(stft, win_length=win_length, hop_length=hop_length)
        if i != num_iters - 1:
            stft = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            _, phase = librosa.magphase(stft)
            phase = np.angle(phase)
            stft = mag * np.exp(1.j * phase)
    return wav


def preemphasis(wav, coeff=0.97):
    """
    Emphasize high frequency range of the waveform by increasing power(squared amplitude).
    Parameters
    ----------
    wav : np.ndarray [shape=(n,)]
        Real-valued the waveform.
    coeff: float <= 1 [scalar]
        Coefficient of pre-emphasis.
    Returns
    -------
    preem_wav : np.ndarray [shape=(n,)]
        The pre-emphasized waveform.
    """
    preem_wav = signal.lfilter([1, -coeff], [1], wav)
    return preem_wav


def inv_preemphasis(preem_wav, coeff=0.97):
    """
    Invert the pre-emphasized waveform to the original waveform.
    Parameters
    ----------
    preem_wav : np.ndarray [shape=(n,)]
        The pre-emphasized waveform.
    coeff: float <= 1 [scalar]
        Coefficient of pre-emphasis.
    Returns
    -------
    wav : np.ndarray [shape=(n,)]
        Real-valued the waveform.
    """
    wav = signal.lfilter([1], [1, -coeff], preem_wav)
    return wav


def linear_to_mel(linear, sr, n_fft, n_mels, **kwargs):
    """
    Convert a linear-spectrogram to mel-spectrogram.
    :param linear: Linear-spectrogram.
    :param sr: Sample rate.
    :param n_fft: FFT window size.
    :param n_mels: Number of mel filters.
    :return: Mel-spectrogram.
    """
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels, **kwargs)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, linear)  # (n_mels, t) # mel spectrogram
    return mel


def amp2db(amp):
    return librosa.amplitude_to_db(amp)


def db2amp(db):
    return librosa.db_to_amplitude(db)


def normalize_db(db, max_db, min_db):
    """
    Normalize dB-scaled spectrogram values to be in range of 0~1.
    :param db: Decibel-scaled spectrogram.
    :param max_db: Maximum dB.
    :param min_db: Minimum dB.
    :return: Normalized spectrogram.
    """
    norm_db = np.clip((db - min_db) / (max_db - min_db), 0, 1)
    return norm_db


def denormalize_db(norm_db, max_db, min_db):
    """
    Denormalize the normalized values to be original dB-scaled value.
    :param norm_db: Normalized spectrogram.
    :param max_db: Maximum dB.
    :param min_db: Minimum dB.
    :return: Decibel-scaled spectrogram.
    """
    db = np.clip(norm_db, 0, 1) * (max_db - min_db) + min_db
    return db


def dynamic_range_compression(db, threshold, ratio, method='downward'):
    """
    Execute dynamic range compression(https://en.wikipedia.org/wiki/Dynamic_range_compression) to dB.
    :param db: Decibel-scaled magnitudes
    :param threshold: Threshold dB
    :param ratio: Compression ratio.
    :param method: Downward or upward.
    :return: Range compressed dB-scaled magnitudes
    """
    if method is 'downward':
        db[db > threshold] = (db[db > threshold] - threshold) / ratio + threshold
    elif method is 'upward':
        db[db < threshold] = threshold - ((threshold - db[db < threshold]) / ratio)
    return db


def emphasize_magnitude(mag, power=1.2):
    """
    Emphasize a magnitude spectrogram by applying power function. This is used for removing noise.
    :param mag: magnitude spectrogram.
    :param power: exponent.
    :return: emphasized magnitude spectrogram.
    """
    emphasized_mag = np.power(mag, power)
    return emphasized_mag


def wav2melspec(wav, sr, n_fft, win_length, hop_length, n_mels, time_first=True, **kwargs):
    # Linear spectrogram
    mag_spec, phase_spec = wav2spec(wav, n_fft, win_length, hop_length, time_first=False)

    # Mel-spectrogram
    mel_spec = linear_to_mel(mag_spec, sr, n_fft, n_mels, **kwargs)

    # Time-axis first
    if time_first:
        mel_spec = mel_spec.T  # (t, n_mels)

    return mel_spec


def wav2melspec_db(wav, sr, n_fft, win_length, hop_length, n_mels, normalize=False, max_db=None, min_db=None,
                   time_first=True, **kwargs):
    # Mel-spectrogram
    mel_spec = wav2melspec(wav, sr, n_fft, win_length, hop_length, n_mels, time_first=False, **kwargs)

    # Decibel
    mel_db = librosa.amplitude_to_db(mel_spec)

    # Normalization
    mel_db = normalize_db(mel_db, max_db, min_db) if normalize else mel_db

    # Time-axis first
    if time_first:
        mel_db = mel_db.T  # (t, n_mels)

    return mel_db


def wav2mfcc(wav, sr, n_fft, win_length, hop_length, n_mels, n_mfccs, preemphasis_coeff=0.97, time_first=True,
             **kwargs):
    # Pre-emphasis
    wav_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Decibel-scaled mel-spectrogram
    mel_db = wav2melspec_db(wav_preem, sr, n_fft, win_length, hop_length, n_mels, time_first=False, **kwargs)

    # MFCCs
    mfccs = np.dot(librosa.filters.dct(n_mfccs, mel_db.shape[0]), mel_db)

    # Time-axis first
    if time_first:
        mfccs = mfccs.T  # (t, n_mfccs)

    return mfccs

def Convert(pred_spec, y_spec, ppgs):

    # Denormalizatoin
    pred_spec = denormalize_db(pred_spec, hp.Default.max_db, hp.Default.min_db)
    y_spec = denormalize_db(y_spec, hp.Default.max_db, hp.Default.min_db)
    #heatmap = np.expand_dims(pred_spec, 3)
    #tf.summary.image('pred_spec_Denormalization', heatmap, max_outputs=pred_spec.shape[0])

    # Db to amp
    pred_spec = db2amp(pred_spec)
    y_spec = db2amp(y_spec)
    #heatmap2 = np.expand_dims(pred_spec, 3)
    #tf.summary.image('pred_spec_Denormalization_Db2amp', heatmap2, max_outputs=pred_spec.shape[0])

    # Emphasize the magnitude
    pred_spec = np.power(pred_spec, hp.Convert.emphasis_magnitude)
    #heatmap3 = np.expand_dims(pred_spec, 3)
    #tf.summary.image('pred_spec_Denormalization_Db2amp_emphasize', heatmap3, max_outputs=pred_spec.shape[0])
    y_spec = np.power(y_spec, hp.Convert.emphasis_magnitude)

    # Spectrogram to waveform
    audioWavList = []
    y_audioWavList = []
    for i in range(len(pred_spec)):
        audioWavList.append(spec2wav(pred_spec[i].T, hp.Default.n_fft, hp.Default.win_length, hp.Default.hop_length,hp.Default.n_iter))
        y_audioWavList.append(spec2wav(y_spec[i].T, hp.Default.n_fft, hp.Default.win_length, hp.Default.hop_length,hp.Default.n_iter))
    audio = np.array(audioWavList)
    y_audio = np.array(y_audioWavList)

    #tf.summary.audio('preemphasisB', audio, hp.Default.sr, max_outputs=hp.Convert.batch_size)
    #tf.summary.audio('preemphasisA', y_audio, hp.Default.sr, max_outputs=hp.Convert.batch_size)

    # Apply inverse pre-emphasis
    audio = inv_preemphasis(audio, coeff=hp.Default.preemphasis)
    y_audio = inv_preemphasis(y_audio, coeff=hp.Default.preemphasis)

    print(audio.shape)

    return audio, y_audio, ppgs


def Convert2(pred_spec):

    # Denormalizatoin
    pred_spec = denormalize_db(pred_spec, 40, -50)

    # Db to amp
    pred_spec = db2amp(pred_spec)

    # Emphasize the magnitude
    pred_spec = np.power(pred_spec, 1.6)

    # Spectrogram to waveform
    audioWavList = []
    for i in range(len(pred_spec)):
        audioWavList.append(spec2wav(pred_spec[i].T, 1136, 1136, 96,60))
    audio = np.array(audioWavList)

    # Apply inverse pre-emphasis
    audio = inv_preemphasis(audio, coeff=0.97)

    return audio

def normalize_0_1(values, max, min):
    normalized = np.clip((values - min) / (max - min), 0, 1)
    '''
     Clip (limit) the values in an array.
     Examples
        --------
        >>> a = np.arange(10)
        >>> np.clip(a, 1, 8)
        array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        >>> a
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> np.clip(a, 3, 6, out=a)
        array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
        >>> a = np.arange(10)
        >>> a
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> np.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
        array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])
    '''
    return normalized

def denormalize_0_1(normalized, max, min):
    values =  np.clip(normalized, 0, 1) * (max - min) + min
    return values


if __name__=='__main__':
    beg = time.time()
    # pred_spec = np.random.rand(1,334,569)
    pred_spec = np.load('C:/Users/LT/Desktop/source.npy')# .reshape(1, 334, 569)
    Convert2(pred_spec)
    print('time:',time.time()-beg)