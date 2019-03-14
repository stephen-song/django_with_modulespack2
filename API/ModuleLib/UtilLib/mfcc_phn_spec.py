import librosa
import numpy as np

from API.ModuleLib.UtilLib import params as hp
from API.ModuleLib.UtilLib.audio import read_wav,preemphasis,amp2db,normalize_0_1

phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']


def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn


def wav_random_crop(wav, sr, duration):
    assert (wav.ndim <= 2)

    target_len = sr * duration
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def get_mfccs_and_phones(wav_file, trim=False, random_crop=True):

    '''This is applied in `train1` or `test1` phase.
    '''

    # Load
    wav = read_wav(wav_file, sr=hp.Default.sr)

    mfccs, _, _ = _get_mfcc_and_spec(wav, hp.Default.preemphasis, hp.Default.n_fft,
                                     hp.Default.win_length,
                                     hp.Default.hop_length)

    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    phn_file = wav_file.replace("_.wav", ".PHN")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // hp.Default.hop_length
        phns[bnd:] = phn2idx[phn]
        bnd_list.append(bnd)

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Random crop
    n_timesteps = (hp.Default.duration * hp.Default.sr) // hp.Default.hop_length + 1
    if random_crop:
        start = np.random.choice(range(np.maximum(1, len(mfccs) - n_timesteps)), 1)[0]
        end = start + n_timesteps
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, n_timesteps, axis=0)
    phns = librosa.util.fix_length(phns, n_timesteps, axis=0)


    return mfccs, phns
    #mfccs==>data，phns==>label

def get_mfccs_and_spectrogram(wav_file, trim=True, random_crop=False):
    '''This is applied in `train2`, `test2` or `convert` phase.
    '''

    # Load
    wav, _ = librosa.load(wav_file, sr=hp.Default.sr)

    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav, frame_length=hp.Default.win_length, hop_length=hp.Default.hop_length)


    if random_crop:
        wav = wav_random_crop(wav, hp.Default.sr, hp.Default.duration)

    # Padding or crop
    length = hp.Default.sr * hp.Default.duration
    wav = librosa.util.fix_length(wav, length)

    return _get_mfcc_and_spec(wav, hp.Default.preemphasis, hp.Default.n_fft, hp.Default.win_length, hp.Default.hop_length)

# TODO refactoring
def _get_mfcc_and_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):

    # Pre-emphasis 预加重 消除发声过程中，声带和嘴唇造成的效应，来补偿语音信号受到发音系统所压抑的高频部分。并且能突显高频的共振峰。
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram  Short-time Fourier transform (STFT)
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # print('D:\n',D)

    #magnitude（振幅）
    mag = np.abs(D)
    # print('mag:\n',mag)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.Default.sr, hp.Default.n_fft, hp.Default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram
    # print('mel:\n',mel)

    # Get mfccs, amp to db
    mag_db = amp2db(mag)
    mel_db = amp2db(mel)
    mfccs = np.dot(librosa.filters.dct(hp.Default.n_mfcc, mel_db.shape[0]), mel_db)

    # 修改：不对mag和mel进行正规化
    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, hp.Default.max_db, hp.Default.min_db)
    mel_db = normalize_0_1(mel_db, hp.Default.max_db, hp.Default.min_db)
    # print('mag_db:\n',mag_db)
    # print('mel_db:\n',mel_db)

    return mfccs.T, mag_db.T, mel_db.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)
    #      x_mfcc , y_spec  , y_mel

if __name__=='__main__':
    mfcc,spec,mel = get_mfccs_and_spectrogram('C:/Users/LT/Desktop/source.wav')
    print(spec)
    np.save('C:/Users/LT/Desktop/source.npy',spec)