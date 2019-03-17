from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse,HttpResponse
from multiprocessing import Process,Pool


# import numpy as np
# import tensorflow as tf
# import os,uuid,io,json,soundfile
# from PIL import Image
# from base64 import b64decode,b64encode
# # Communication to TensorFlow server via gRPC
# from grpc.beta import implementations
# # TensorFlow serving stuff to send messages
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2
# from tensorflow.contrib.util import make_tensor_proto

import time,os,json,copy
from API.ModulesPack2.session.base import Session
from API.GraphLib import get_graph

# Create your views here.
@csrf_exempt
def home(request):
    return render(request, 'classify.html', {})

@csrf_exempt
def test(request):
    time.sleep(3)
    return HttpResponse('OK')

begin = time.time()
print('start init')
graph = get_graph('VC')
sess = Session(ModuleLibPath=os.path.expanduser('~/0307_django/API/ModuleLib')) # 'D:\pycharm_proj/0307_django\API\ModuleLib'
ModuleTable, toposort = sess.build_graph(graph)
print('outside init time:',time.time()-begin)


# pool = Pool(processes=10)
# def service_target(sess,beg,request,fetches,ModuleTable, graph):
#     print('Process : %s' % os.getpid())
#     beg2 = time.time()
#
#     feed_dic = {'request_handler': {'request': request}}
#     result = copy.deepcopy(sess).run(fetches=fetches,
#                                      ModuleTable=copy.deepcopy(ModuleTable)
#                                      , graph=graph, feed_dict=feed_dic)
#
#     beg3 = time.time()
#     print('run', beg3 - beg2)
#     print('total', time.time() - beg)
#     return result['RetDataDict']

@csrf_exempt
def API(request):
    print('father Process : %s' % os.getpid())
    service_type = 'test'
    beg = time.time()
    if request.method == "POST":
        if request.POST.get("service", None) is not None:
            service_type = request.POST.get("service", None).strip()  # VC(数据)
            # print('\033[1;33;44m inner {} \033[0m'.format(service_type))
        else:
            received_json_data = json.loads(request.body.decode('utf-8'))
            service_type = received_json_data["service"]

    elif request.method == "GET":
        if request.GET.get("service", None) is not None:
            service_type = request.GET.get("service", None).strip()  # VC(数据)
            # print('\033[1;33;44m inner {} \033[0m'.format(service_type))
        else:
            received_json_data = json.loads(request.body.decode('utf-8'))
            service_type = received_json_data["service"]
    else:
        service_type = 'test'
        # print('\033[1;33;44m inner {} \033[0m'.format(service_type))

    print('\033[1;33;44m {} \033[0m'.format(service_type))
    # 把初始化放在了外面
    # graph = get_graph(service_type)
    # sess = Session(ModuleLibPath=os.path.expanduser('~/0307_django/API/ModuleLib')) # 'D:\pycharm_proj/0307_django\API\ModuleLib'
    # ModuleTable, toposort = sess.build_graph(graph)

    beg2 = time.time()
    print('init_sess_graph',beg2-beg)

    feed_dic = {'request_handler': {'request': request}}
    result = copy.deepcopy(sess).run(fetches=toposort,ModuleTable=copy.deepcopy(ModuleTable)
                    ,graph=graph, feed_dict=feed_dic)

    beg3 = time.time()
    print('run',beg3-beg2)
    print('total',time.time() - beg)

    return JsonResponse(result['RetDataDict'])


    # result = pool.apply_async(service_target,args=(sess,beg,request,toposort,ModuleTable,graph))
    # while 1:
    #     if result.successful():
    #         response = JsonResponse(result.get())
    #         break
    #
    # return response