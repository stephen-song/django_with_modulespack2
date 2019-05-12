import time,os
beg =time.time()
from ModulesPack2.session.base import Session

from ModulesPack2.graph.base import ModuleGraph

graph = ModuleGraph(JsonFile='GraphLib/test.json')
ModuleLibPath = 'D:\pycharm_proj/0307_django\API\ModuleLib' # os.path.expanduser('~/0307_django/API/ModuleLib')
sess = Session(ModuleLibPath=ModuleLibPath)
feed_dic = {'_test':{'x1':1,'x2':2}}

ModuleTable,toposort = sess.build_graph(graph)
result =               sess.run(fetches=toposort,ModuleTable=ModuleTable
                                ,graph=graph,feed_dict=feed_dic)
print(result)
print(time.time()-beg)
# graph.ShowGraph()
