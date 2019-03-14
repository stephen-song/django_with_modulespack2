import time,librosa
from scipy import signal
import numpy as np
# from API.ModuleLib.UtilLib.mfcc_phn_spec import get_mfccs_and_spectrogram
from API.ModulesPack2.module.base import Module
from API.ModulesPack2.module.module_desc import ModuleDesc, InputDesc, OutputDesc

__all__ = ['Mfcc_Spec_Mel']
class Mfcc_Spec_Mel(Module):
    @staticmethod
    def make_module_description():
        inputdesc = {
            'wav_file': InputDesc(datatype='string', datashape=(None,))
        }
        outputdesc = {
            'mfcc': OutputDesc(datatype='np.float32', datashape=(334, 60)),
            'spec': OutputDesc(datatype='np.float32',datashape=(334, 569)),
            'mel': OutputDesc(datatype='np.float32',datashape=(334, 90))
        }
        MD = ModuleDesc(inputdesc, outputdesc)
        return MD
    @staticmethod
    def run(inputs):
        beg =time.time()
        # 获取输入
        wav_file = inputs['wav_file']

        # 处理数据  # os.remove(wav_file)
        wav, _ = librosa.load(wav_file, sr=16000)# Load
        wav, _ = librosa.effects.trim(wav, frame_length=1136, hop_length=96)# Trim
        length = 16000 * 2
        wav = librosa.util.fix_length(wav, length)# Padding or crop
        y_preem = preemphasis(wav, coeff=0.97)# Pre-emphasis
        D = librosa.stft(y=y_preem, n_fft=1136, hop_length=96, win_length=1136)# Get spectrogram  Short-time Fourier transform (STFT)
        mag = np.abs(D)# magnitude（振幅）
        mel_basis = librosa.filters.mel(16000, 1136, 90)  # (n_mels, 1+n_fft//2) # Get mel-spectrogram
        mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram
        mag_db = amp2db(mag)
        mel_db = amp2db(mel)
        mfccs = np.dot(librosa.filters.dct(60, mel_db.shape[0]), mel_db)# Get mfccs, amp to db
        mag_db = normalize_0_1(mag_db, 40, -50)# Normalization (0 ~ 1)
        mel_db = normalize_0_1(mel_db, 40, -50)
        x_mfcc, y_spec, y_mel = mfccs.T, mag_db.T, mel_db.T

        # 返回
        ret = { 'mfcc': x_mfcc, 'spec' : y_spec, 'mel' : y_mel }
        print('Mfcc_Spec_Mel Time:', time.time()-beg)
        return ret


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

def amp2db(amp):
    return librosa.amplitude_to_db(amp)

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

if __name__=='__main__':
    test = Mfcc_Spec_Mel()

    inputs = {'wav_file': 'C:/Users/LT/Desktop/source.wav'} #'C:/Users/LT/AppData/Local/Temp/tmp_urim1rc.wav'

    print(test.run(inputs))