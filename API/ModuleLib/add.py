from API.ModulesPack2.module.base import Module
from API.ModulesPack2.module.module_desc import ModuleDesc,InputDesc,OutputDesc

__all__ = ['add']
class add(Module):

    def __init__(self):
        pass

    @staticmethod
    def make_module_description():
        inputdesc = {'x1':InputDesc(datatype='np.float32',datashape=(None,)),
                     'x2':InputDesc(datatype='np.float32',datashape=(None,))}

        outputdesc = {'y':OutputDesc(datatype='np.float32',datashape=(None,))}

        MD = ModuleDesc(inputdesc,outputdesc)
        return MD

    @staticmethod
    def run(inputs):
        num1 = inputs['x1']
        num2 = inputs['x2']
        ret = {'y':num1+num2}
        return ret

if __name__=='__main__':
    a = add()
    inputs={'x1':1,'x2':2}
    print(a.run(inputs))