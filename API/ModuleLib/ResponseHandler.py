from base64 import b64encode

from API.ModulesPack2.module.base import Module
from API.ModulesPack2.module.module_desc import ModuleDesc,InputDesc,OutputDesc
import time

__all__ = ['ResponseHandler']
class ResponseHandler(Module):

    @staticmethod
    def make_module_description():
        inputdesc = {
            'data': InputDesc(datatype='string', datashape=(None,)),
            'return_type': InputDesc(datatype='string', datashape=(None,))
        }
        outputdesc = {'RetDataDict': OutputDesc(datatype='{}', datashape=(None,))}
        MD = ModuleDesc(inputdesc, outputdesc)
        return MD

    @staticmethod
    def run(inputs):
        beg =time.time()
        # 获取输入
        data = inputs['data']
        return_type = inputs['return_type']

        if return_type == 'audio':
            # 处理数据
            stream = b64encode(data)
            # base64 to str
            stream = str(stream, 'utf-8')
            stream = 'data:audio/wav;base64,' + stream

        elif return_type == 'text':
            stream = data

        # 返回
        DataDict = {'result':stream, 'success': True}
        ret = { 'RetDataDict': DataDict }
        print('ResponseHandler Time:', time.time() - beg)
        return ret