from tempfile import TemporaryFile
from tempfile import NamedTemporaryFile
import numpy as np
import os,soundfile
from base64 import b64encode
# # 关闭文件时候删除
# f = TemporaryFile()
#
# # delete默认删除，为True则关闭临时文件时候不删除，
# f_2 = NamedTemporaryFile(delete=False)
#
# f.write(b'abcd' * 100)
# f_2.write(b'abcd' * 100)
#
# # 并不能自主命名。系统分配名字，只能写入bytes类型
# print(f_2.name, f.name)
# # print(f_2.name.type,f.name.type)
# print(type(''))

class decodeData():

    @staticmethod
    def d():
        print('decodeData')

audio = np.random.rand(1000,1)
fp = NamedTemporaryFile(delete=False)
soundfile.write(file=fp, data=audio.T, samplerate=16000,format='WAV')
fp.close()
with open(fp.name, 'rb') as audio_file:
    stream = audio_file.read()
    # bytes to base64
    # stream = b64encode(stream)
    # base64 to str
    # stream = str(stream, 'utf-8')
    # stream = 'data:audio/wav;base64,' + stream
    audio_file.close()
os.remove(fp.name)
print(stream)