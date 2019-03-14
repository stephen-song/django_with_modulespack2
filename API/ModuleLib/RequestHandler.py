import json,time
import tempfile
from base64 import b64decode

from API.ModulesPack2.module.base import Module
from API.ModulesPack2.module.module_desc import ModuleDesc,InputDesc,OutputDesc

__all__ = ['RequestHandler']
class RequestHandler(Module):

    @staticmethod
    def make_module_description():
        inputdesc = { 'request': InputDesc(datatype='HttpRequest', datashape=(None,)) }
        outputdesc = {'data_file_path': OutputDesc(datatype='string', datashape=(None,))}
        MD = ModuleDesc(inputdesc, outputdesc)
        return MD

    @staticmethod
    def run(inputs):
        beg = time.time()
        # 获取输入
        request = inputs['request']

        # 处理数据
        if request.method == 'POST':
            QueryDict = request.POST
        elif request.method == 'GET':
            QueryDict = request.GET
        else:
            raise Exception('request method must be POST or GET!')

        if QueryDict.get("data") is not None:
            # 得到的是经过b64encode的编码字符串: 'YmluYXJ5AHN0cmluZw=='  # 因为js文件中readAsDataURL会使用base64进行编码
            contents = QueryDict.get("data")  # data:audio/wav;base64,UklGRuT5...(数据)
            bytes_data = contents.split(',', 1)[1]  # UklGRuT5...(数据)
            extensions = contents.split(',')[0].rsplit('/')[1].split(';')  # ['wav','base64']
            # 转换为字符串未编码时的内容: 'binary\x00string'
            plain_data = b64decode(bytes_data)
            # print('\033[1;33;44m QueryDict data \033[0m')
        else:
            # body为json格式
            received_json_data = json.loads(request.body.decode('utf-8'))
            # print(type(received_json_data))
            data = received_json_data["data"]
            # print(type(data))
            plain_data = b64decode(bytes(data,encoding='utf8'))
            # print('\033[1;33;44m json data \033[0m')

        fp = tempfile.NamedTemporaryFile(delete=False,suffix='.wav')
        fp.write(plain_data)
        fp.close()

        # 返回
        ret = { 'data_file_path': fp.name }
        print('RequestHandler Time:',time.time()-beg)
        return ret