from API.ModulesPack2.module.base import Module
from API.ModulesPack2.module.module_desc import ModuleDesc,InputDesc,OutputDesc
from API.GraphLib import get_graph
import os,time

__all__ = ['Voice_Conversion']


graph = get_graph('_VC')
from API.ModulesPack2.session.base import Session
sess = Session(ModuleLibPath=os.path.expanduser('~/0307_django/API/ModuleLib'))
ModuleTable, toposort = sess.build_graph(graph)

class Voice_Conversion(Module):

    @staticmethod
    def make_module_description():
        inputdesc = {
            'wav_file_path': InputDesc(datatype='string', datashape=(None,))
        }
        outputdesc = {
            'audio_bytes': OutputDesc(datatype='string', datashape=(None,))
        }
        MD = ModuleDesc(inputdesc, outputdesc)
        return MD

    @staticmethod
    def run(inputs):
        beg = time.time()
        # 获取输入
        wav_file = inputs['wav_file_path']

        # 处理数据
        # graph = get_graph('_VC')
        # from API.ModulesPack2.session.base import Session
        # sess = Session(ModuleLibPath=os.path.expanduser('~/0307_django/API/ModuleLib'))
        # ModuleTable, toposort = sess.build_graph(graph)

        feed_dic = {'mfcc_spec_mel': {'wav_file': wav_file}}
        result = sess.run(fetches=toposort, ModuleTable=ModuleTable
                          , graph=graph, feed_dict=feed_dic)


        # 返回
        ret = { 'audio_bytes': result['audio_stream'] }
        print('Voice_Conversion Time:' ,time.time() - beg )
        return ret