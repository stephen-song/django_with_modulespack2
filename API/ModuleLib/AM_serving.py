import numpy as np
# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import librosa
from keras import backend as K

from API.ModulesPack2.module.base import Module
from API.ModulesPack2.module.module_desc import ModuleDesc,InputDesc,OutputDesc
import time,os
from API.ModuleLib import ModuleLibPath

__all__ = ['AM_serving']

server = '219.223.172.28:9001'
model_name = 'ASR_am'
am_vocab = np.load(os.path.join(ModuleLibPath,'UtilLib/am_pinyin_dict.npy')).tolist()

def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = result.shape[1]
	r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text

class AM_serving(Module):
    @staticmethod
    def make_module_description():
        inputdesc = {
            'mfcc': InputDesc(datatype='np.float32', datashape=(None, None, 26)),
            'len': InputDesc(datatype='np.int32', datashape=(None,1)),
        }
        outputdesc = {
            'PinYin': OutputDesc(datatype='list', datashape=(None,)),
        }
        MD = ModuleDesc(inputdesc, outputdesc)
        return MD

    @staticmethod
    def run(inputs):
        beg = time.time()
        # 获取输入
        mfcc = inputs['mfcc']
        len = inputs['len']

        # 处理数据
        host, port = server.split(':')
        channel = implementations.insecure_channel(host, int(port))
        # stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel._channel)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.signature_name = 'predict_AudioSpec2Pinyin'
        request.inputs['mfcc'].CopyFrom(make_tensor_proto(mfcc, shape=mfcc.shape, dtype='float'))  # 1是batch_size
        request.inputs['len'].CopyFrom(make_tensor_proto(len, shape=len.shape, dtype='int32'))
        # print('begin')
        result = stub.Predict(request, 60.0)

        # pred_logits = np.array(result.outputs['logits'].float_val).reshape(1, -1, len(am_vocab)).astype(np.float32)
        labels = np.asarray(result.outputs['label'].int_val)
        # print('label.shape:', labels.shape)
        # print('label      :', labels)
        # print('pred_logits:', pred_logits)

        # r1, pinyin = decode_ctc(pred_logits, am_vocab)
        # print('pred_label     :', pinyin)

        pinyin = [am_vocab[i] for i in labels]
        # print('true_label     :', pinyin)

        # 返回
        ret = {'PinYin': pinyin}
        print('AM_serving Time:', time.time() - beg)
        return ret