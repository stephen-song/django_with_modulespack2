import requests
import time
import base64
import threading
import json


def main():
    benchmark(2)
    time.sleep(3)

    # benchmark2(2)
    # time.sleep(10)
    #
    # benchmark2(2)
    # time.sleep(10)

    # benchmark(50)
    # time.sleep(3)
    #
    # benchmark(100)
    # time.sleep(3)

end_point = 'http://219.223.174.254:8080/API/' # THU/VC_api
audio_64_string = base64.b64encode(open("source.wav","rb").read())
input_data = {"data":audio_64_string.decode('utf-8'),"service":"VC"}
# input_data = {"data":audio_64_string.decode('utf-8')[0:10]}
headers = {'User-Agent' : 'Mozilla/5.0 (Linux; Android 4.2.1; en-us; Nexus 4 Build/JOP40D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Mobile Safari/535.19'}
error_result = []
correct_result = []
def target():
    try:
        result = requests.post(end_point, json=input_data,headers=headers)  #这里不需要dump
        ResponseTime = float(result.elapsed.seconds) # 获取响应时间，单位s
        if result.status_code != 200:
            error_result.append("0")
        else:
            correct_result.append(ResponseTime)
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



def benchmark2(batch_size):
    error_result.clear()
    correct_result.clear()
    beg = time.time()
    for i in range(batch_size):
        try:
            result = requests.post(end_point, json=input_data, headers=headers)  # 这里不需要dump
            ResponseTime = float(result.elapsed.seconds)  # 获取响应时间，单位s
            if result.status_code != 200:
                error_result.append("0")
            else:
                correct_result.append(ResponseTime)
                print(result)
        except Exception as e:
            print(e)
    end = time.time()
    print('concurrent processing: ', batch_size)
    print('failed request:', len(error_result))
    print('correct_result_time: ', correct_result)
    print('avg_response time: ', float(sum(correct_result)) / float(len(correct_result)))
    print('total time: ', (end - beg), ' s')
    print('seconds  per completed request: ', (end - beg) * 1. / batch_size * 1., ' s/request')
    print('吞吐率 requests  per second: ', batch_size * 1. / (end - beg) * 1., ' request/s')


if __name__=='__main__':
    main()
    # print('b64encode: ',audio_64_string)
    # print()
    # print()
    # print()
    # print('utf-8: ',audio_64_string.decode('utf-8'))
    # a = bytes(audio_64_string.decode('utf-8'),encoding='utf8')
    # if a==audio_64_string:
    #     print('OK')
    # else:
    #     print('utf8-->base64: ',a)

