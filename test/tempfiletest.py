from tempfile import NamedTemporaryFile
import base64
fp = NamedTemporaryFile(delete=False,suffix='.wav')

fp.write(b'aaa')
fp.close()
print(fp.name)