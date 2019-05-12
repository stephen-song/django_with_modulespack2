# import tensorflow as tf  # import 严重影响速度
import numpy as np
# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from API.ModulesPack2.module.base import Module
from API.ModulesPack2.module.module_desc import ModuleDesc,InputDesc,OutputDesc
import time
from random import choice

__all__ = ['VC_serving']

tf_serving_server = [
    {'server':'219.223.172.28:9003','name':'VC2.1'},
    {'server': '219.223.172.28:9004', 'name': 'VC2.2'}
]
indexs = [0,1]

class VC_serving(Module):
    @staticmethod
    def make_module_description():
        inputdesc = {
            'mfcc': InputDesc(datatype='np.float32', datashape=(334, 60)),
            'spec': InputDesc(datatype='np.float32',datashape=(334, 569)),
            'mel': InputDesc(datatype='np.float32',datashape=(334, 90))
        }
        outputdesc = {
            'ppgs': OutputDesc(datatype='np.float32', datashape=(1, 334, 61)),
            'pred_spec': OutputDesc(datatype='np.float32',datashape=(1, 334, 569))
        }
        MD = ModuleDesc(inputdesc, outputdesc)
        return MD
    @staticmethod
    def run(inputs):
        beg = time.time()
        # 获取输入
        mfcc = inputs['mfcc']
        spec = inputs['spec']
        mel = inputs['mel']

        # 处理数据
        index = choice(indexs)
        print('index:',index)
        server = tf_serving_server[index]['server']
        host, port = server.split(':')
        channel = implementations.insecure_channel(host, int(port))
        # stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel._channel)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = tf_serving_server[index]['name']
        request.model_spec.signature_name = 'prediction_pipeline'
        request.inputs['x_mfccs:0'].CopyFrom(
            make_tensor_proto(mfcc, shape=[1, 334, 60], dtype='float'))  # 1是batch_size
        request.inputs['y_spec:0'].CopyFrom(
            make_tensor_proto(spec, shape=[1, 334, 569], dtype='float'))
        request.inputs['y_mel:0'].CopyFrom(
            make_tensor_proto(mel, shape=[1, 334, 90], dtype='float'))

        result=stub.Predict(request, 60.0)
        pred_spec=np.array(result.outputs['pred_spec:0'].float_val).reshape(1, 334, 569)
        ppgs=np.array(result.outputs['ppgs:0'].float_val).reshape(1, 334, 61)
        # np.save('source.npy',pred_spec)

        # 返回
        ret = { 'pred_spec': pred_spec , 'ppgs' : ppgs }
        print('VC_serving Time:',time.time()-beg)
        return ret