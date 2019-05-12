from API.ModulesPack2.module.base import Module
from API.ModulesPack2.module.module_desc import ModuleDesc,InputDesc,OutputDesc
from API.GraphLib import get_graph
import os,time
from API.ModuleLib import ModuleLibPath

__all__ = ['Automatic_Speech_Recognition']


asr_graph = get_graph('_ASR')
from API.ModulesPack2.session.base import Session
asr_sess = Session(ModuleLibPath=ModuleLibPath)
asr_ModuleTable, asr_toposort = asr_sess.build_graph(asr_graph)

class Automatic_Speech_Recognition(Module):

    @staticmethod
    def make_module_description():
        inputdesc = {
            'wav_file_path': InputDesc(datatype='string', datashape=(None,))
        }
        outputdesc = {
            'text': OutputDesc(datatype='string', datashape=(None,))
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

        feed_dic = {'audio2mfcc': {'wav_file': wav_file}}
        result = asr_sess.run(fetches=asr_toposort, ModuleTable=asr_ModuleTable
                          , graph=asr_graph, feed_dict=feed_dic)


        # 返回
        ret = { 'text': result['HanZi'] }
        print('Automatic_Speech_Recognition Time:' ,time.time() - beg )
        return ret