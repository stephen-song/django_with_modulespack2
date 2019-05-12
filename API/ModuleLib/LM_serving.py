import numpy as np
# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from API.ModulesPack2.module.base import Module
from API.ModulesPack2.module.module_desc import ModuleDesc,InputDesc,OutputDesc
import time,os
from API.ModuleLib import ModuleLibPath

__all__ = ['LM_serving']

server = '219.223.172.28:9002'
model_name = 'ASR_lm'
lm_pinyin_vocab = np.load(os.path.join(ModuleLibPath,'UtilLib/lm_pinyin_dict.npy')).tolist()
lm_hanzi_vocab = np.load(os.path.join(ModuleLibPath,'UtilLib/lm_hanzi_dict.npy')).tolist()


class LM_serving(Module):
    @staticmethod
    def make_module_description():
        inputdesc = {
            'PinYin': InputDesc(datatype='list', datashape=(None,)),
        }
        outputdesc = {
            'HanZi': OutputDesc(datatype='string', datashape=(None,)),
        }
        MD = ModuleDesc(inputdesc, outputdesc)
        return MD

    @staticmethod
    def run(inputs):
        beg = time.time()
        # 获取输入
        pinyin = inputs['PinYin']


        # 处理数据
        # pinyin = ['jin1','tian1','tian1','qi4','zhen1','hao3']
        pinyin = [lm_pinyin_vocab.index(i) for i in pinyin]
        pinyin = np.asarray(pinyin).reshape(1, -1)
        # print(pinyin.shape, pinyin)

        host, port = server.split(':')
        channel = implementations.insecure_channel(host, int(port))
        # stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel._channel)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.signature_name = 'predict_Pinyin2Hanzi'
        request.inputs['pinyin'].CopyFrom(
            make_tensor_proto(pinyin, shape=pinyin.shape, dtype='int32'))  # 1是batch_size
        result = stub.Predict(request, 60.0)
        pred_label = np.array(result.outputs['hanzi'].int_val)

        # print(pred_label.shape, pred_label)
        hanzi = [lm_hanzi_vocab[i] for i in pred_label]
        # print( bytes("".join(hanzi), encoding = "utf8"))

        # 返回
        ret = {'HanZi': "".join(hanzi)}
        print('LM_serving Time:', time.time() - beg)
        return ret