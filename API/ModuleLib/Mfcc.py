import time,librosa
import numpy as np

# from API.ModuleLib.UtilLib.mfcc_phn_spec import get_mfccs_and_spectrogram
from API.ModulesPack2.module.base import Module
from API.ModulesPack2.module.module_desc import ModuleDesc, InputDesc, OutputDesc

__all__ = ['Mfcc']

def load_wav(path):
  return librosa.core.load(path, sr=16000)[0]

def compute_mfcc2(file):
    wav = load_wav(file)
    # mfcc = p.mfcc(wav,numcep=hp.num_mfccs) # n_frames*n_mfcc
    mfcc = librosa.feature.mfcc(wav,sr=16000,n_mfcc=26)  # n_mfcc * n_frames
    n_frames = mfcc.shape[1]
    return (mfcc.T,n_frames)

class Mfcc(Module):
    @staticmethod
    def make_module_description():
        inputdesc = {
            'wav_file': InputDesc(datatype='string', datashape=(None,))
        }
        outputdesc = {
            'mfcc': OutputDesc(datatype='np.float32', datashape=(None, None, 26)),
            'seq_len' : OutputDesc(datatype='np.int32', datashape=(None, 1))
        }
        MD = ModuleDesc(inputdesc, outputdesc)
        return MD

    @staticmethod
    def run(inputs):
        beg =time.time()
        # 获取输入
        wav_file = inputs['wav_file']

        # 处理数据  # os.remove(wav_file)
        mfcc, length = compute_mfcc2(wav_file)
        mfcc = np.expand_dims(mfcc, 0)
        # print(mfcc.shape)
        length = np.asarray(length).reshape(1, 1)
        # print(length.shape, length)  # 313

        # 返回
        ret = { 'mfcc': mfcc,  'seq_len' : length }
        print('Mfcc Time:', time.time()-beg)
        return ret



