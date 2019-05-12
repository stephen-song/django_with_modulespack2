#  -*- coding: utf-8 -*-
#  File: __init__.py
import os
ModuleLibPath = 'D:\pycharm_proj/0307_django\API\ModuleLib' # os.path.expanduser('~/0307_django/API/ModuleLib')
# ModuleLibPath = os.path.expanduser('~/0307_django/API/ModuleLib')

# https://github.com/celery/kombu/blob/7d13f9b95d0b50c94393b962e6def928511bfda6/kombu/__init__.py#L34-L36
STATICA_HACK = False
globals()['kcah_acitats'[::-1].upper()] = False
if STATICA_HACK:
    from .Mfcc_Spec_Mel import Mfcc_Spec_Mel
    from .RequestHandler import RequestHandler
    from .ResponseHandler import ResponseHandler
    from .VC_serving import VC_serving
    from .Vocoder import Vocoder
    from .Test import RMSE
    from .add import add
    from .minus import minus
    from .multiply import multiply
    from .divide import divide

# __all__ = ['Mfcc_Spec_Mel','RequestHandler','ResponseHandler'
#            ,'VC_serving','Vocoder','RMSE','add','minus','divide','multiply']
# __all__ = []
#
# from pkgutil import iter_modules
# import os
#
# _Not_SKIP = ['Module','ModuleDesc','InputDesc','OutputDesc','Session','numpy']
# def _global_import(name):
#     p = __import__(name,globals(),locals(),level=1)
#     lst = p.__all__ if '__all__' in dir(p) else dir(p)
#     # print(dir(p))
#     if lst:
#         del globals()[name]
#         for k in lst:
#             # print(k)
#             if not k.startswith('__'):
#                 globals()[k] = p.__dict__[k]
#                 __all__.append(k)
#         # print(type(eval(name)))
#
#
# _CURR_DIR = os.path.dirname(__file__)
# for _, module_name, _ in iter_modules([_CURR_DIR]):
#     srcpath = os.path.join(_CURR_DIR, module_name + '.py')
#     if not os.path.isfile(srcpath):
#         continue
#     if not module_name.startswith('_'):
#         _global_import(module_name)
