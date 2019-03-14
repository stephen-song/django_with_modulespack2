import os,sys

base_dir = 'D:/pycharm_proj/0307_django/test'
sys.path.append(base_dir)
name = 'decodeData'

def test(name):
    p = __import__(name,globals(),locals(),level=0)

    print(dir(p))
    globals()[name] = p.__dict__[name]

    instance = eval(name)

    instance.d()

test(name)