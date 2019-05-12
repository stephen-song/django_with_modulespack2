from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse,HttpResponse
from django.views.decorators.cache import cache_page
# from multiprocessing import Process,Pool


import time,os,json,copy
from API.ModulesPack2.session.base import Session
from API.GraphLib import get_graph
from API.ModuleLib import ModuleLibPath
# Create your views here.
@csrf_exempt
@cache_page(60*5)
def home(request):
    return render(request, 'home.html', {})

@csrf_exempt
@cache_page(60*5)
def VC(request):
    return render(request, 'VC.html', {})

@csrf_exempt
@cache_page(60*5)
def ASR(request):
    return render(request, 'ASR.html', {})

@csrf_exempt
def test(request):
    time.sleep(3)
    return HttpResponse('OK')

begin = time.time()
print('start init')
graph = get_graph('VC')
sess = Session(ModuleLibPath=ModuleLibPath)
ModuleTable, toposort = sess.build_graph(graph)
graph2 = get_graph('ASR')
sess2 = Session(ModuleLibPath=ModuleLibPath)
ModuleTable2, toposort2 = sess2.build_graph(graph2)
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
    if service_type == 'VC':
        feed_dic = {'request_handler': {'request': request}}

        MT = copy.deepcopy(ModuleTable)
        result = sess.run(fetches=toposort,ModuleTable=MT
                        ,graph=graph, feed_dict=feed_dic)
    elif service_type == 'ASR':
        feed_dic2 = {'request_handler': {'request': request}}

        MT2 = copy.deepcopy(ModuleTable2)
        result = sess2.run(fetches=toposort2, ModuleTable=MT2
                          , graph=graph2, feed_dict=feed_dic2)

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