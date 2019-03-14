import os,time
import tempfile
import numpy as np
import soundfile,librosa
from scipy import signal

# from API.ModuleLib.UtilLib.audio import Convert2
from API.ModulesPack2.module.base import Module
from API.ModulesPack2.module.module_desc import ModuleDesc, InputDesc, OutputDesc

__all__ = ['Vocoder']
class Vocoder(Module):

    @staticmethod
    def make_module_description():
        inputdesc = {
            'pred_spec': InputDesc(datatype='np.float32',datashape=(1, 334, 569))
        }
        outputdesc = {
            'audio_stream': OutputDesc(datatype='string', datashape=(None,))
        }
        MD = ModuleDesc(inputdesc, outputdesc)
        return MD

    @staticmethod
    def run(inputs):
        beg = time.time()
        # 获取输入
        pred_spec = inputs['pred_spec']

        # 处理数据
        pred_spec = denormalize_db(pred_spec, 40, -50)# Denormalizatoin
        pred_spec = db2amp(pred_spec)# Db to amp
        pred_spec = np.power(pred_spec, 1.6)# Emphasize the magnitude
        audioWavList = []# Spectrogram to waveform
        for i in range(len(pred_spec)):
            audioWavList.append(spec2wav(pred_spec[i].T, 1136, 1136, 96, 60))
        audio = np.array(audioWavList)
        audio = inv_preemphasis(audio, coeff=0.97)# Apply inverse pre-emphasis

        # audio = Convert2(pred_spec=pred_spec)
        fp = tempfile.NamedTemporaryFile(delete=False)
        soundfile.write(file=fp, data=audio.T, samplerate=16000,format='WAV')
        fp.close()
        with open(fp.name, 'rb') as audio_file:
            stream = audio_file.read()
            # # bytes to base64
            # stream = b64encode(stream)
            # # base64 to str
            # stream = str(stream, 'utf-8')
            # stream = 'data:audio/wav;base64,' + stream
            audio_file.close()
        os.remove(fp.name)

        # 返回
        ret = { 'audio_stream': stream }
        print('Vocoder Time:' , time.time()-beg)
        return ret

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

def db2amp(db):
    return librosa.db_to_amplitude(db)

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