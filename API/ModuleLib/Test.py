from API.ModulesPack2.module.base import Module
from API.ModulesPack2.module.module_desc import ModuleDesc,InputDesc,OutputDesc
from API.ModulesPack2.graph.base import ModuleGraph
# from ModulesPack2.session import Session

__all__ = ['Test']
class Test(Module):

    def __init__(self):
        pass

    @staticmethod
    def make_module_description():
        inputdesc = {'x1':InputDesc(datatype='np.float32', datashape=(None,)),
                     'x2':InputDesc(datatype='np.float32', datashape=(None,))}
        outputdesc = {'y':OutputDesc(datatype='np.float32', datashape=(None,))}
        MD = ModuleDesc(inputdesc, outputdesc)
        return MD

    @staticmethod
    def run(inputs):
        x1 = inputs['x1']
        x2 = inputs['x2']

        graph = ModuleGraph(JsonFile='GraphLib/_test.json')
        from API.ModulesPack2.session.base import Session
        sess = Session(ModuleLibPath='D:\pycharm_proj/0307_django\API\ModuleLib')
        feed_dic = {'firstadd': {'x1': x1, 'x2': x2}}

        ModuleTable, toposort = sess.build_graph(graph)
        result = sess.run(fetches=toposort, ModuleTable=ModuleTable
                          , graph=graph, feed_dict=feed_dic)

        return result