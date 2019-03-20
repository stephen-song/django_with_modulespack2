from aip import AipSpeech
import time
import threading
""" 你的 APPID AK SK """
APP_ID = '15406461'
API_KEY = 'F6w0hOR3lfeMQGmtY3gaLE2c'
SECRET_KEY = '3FAqhPNmip4VnuimH9Rrea7lv3kGUrWa'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()
speech = get_file_content('source.wav')
# 识别本地文件

error_result = []
correct_result = []
def target():
    try:
        result,ResponseTime = client.asr(speech, 'wav', 16000, {'dev_pid': 1536})
        if result["err_no"] != 0:
            error_result.append("0")
            print(result)
        else:
            correct_result.append(ResponseTime)
            print(result["result"])
    except Exception as e:
        print(e)

def benchmark(batch_size):
    threads =[]
    error_result.clear()
    correct_result.clear()

    beg = time.time()
    try:
        for i in range(batch_size):
            thd = threading.Thread(target=target)
            threads.append(thd)
            # thd.start()

        last_j = None
        for j in threads:
            # j.setDaemon(True) # https://www.cnblogs.com/alan-babyblog/p/5325071.html
            j.start()
            last_j =j
        last_j.join()
    except Exception as e:
        print(e)
    end = time.time()

    print('concurrent processing: ',batch_size)
    print('failed request:', len(error_result))
    print('correct_result_time: ' ,correct_result)
    print('avg_response time: ' ,float(sum(correct_result)) / float(len(correct_result)))
    print('total time: ',(end-beg),' s')
    print('seconds  per completed request: ', (end-beg)*1./batch_size*1.,' s/request')
    print('吞吐率 requests  per second: ', batch_size * 1. / (end - beg) * 1. , ' request/s')

def main():
    # benchmark(1)
    # time.sleep(10)
    #
    # benchmark(2)
    # time.sleep(10)

    benchmark(20)
    # time.sleep(10)

    # benchmark(50)
    # time.sleep(3)
    #
    # benchmark(100)
    # time.sleep(3)

if __name__=='__main__':
    main()
