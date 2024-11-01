import os
import random
import time
import requests
import proccess_gif
import shutil
mode = 0
download = []
if mode == 0:
    with open("b.txt",encoding='utf-8') as a: #训练表情包
        download = a.readlines()
elif mode == 1:
    with open("i.txt",encoding='utf-8') as a:#训练奶龙表情包
        download = a.readlines()
elif mode == 2:
    with open("test_n.txt",encoding='utf-8') as a:#测试正常表情包
        download = a.readlines()
elif mode == 3:
    with open("test_nai.txt",encoding='utf-8') as a:#测试奶龙表情包
        download = a.readlines()


i = 0
headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "zh-CN,zh;q=0.9",
    "priority": "u=0, i",
    "sec-ch-ua": '"Chromium";v="130", "Microsoft Edge";v="130", "Not;A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
    "referer": "https://www.bing.com/" 
}
os.system('cls')
i = 0
i1 = 0
for n_d in download:
    n_d = n_d.strip() 
    print(f"downloading {n_d}")
    try:
        response = requests.get(n_d,headers=headers)
    except Exception as e:
        print("ERROR at",n_d)
        print(f"图片下载失败，状态码：{response.status_code}")
        with open("log.txt","a+",encoding='utf-8') as log:
            log.write(f"at time {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} download file:{n_d} fail! {e} response.status_code:{response.status_code}")
        continue
    if response.status_code == 200:
        # 从URL中提取图片文件名
        image_name = f"{mode}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_gif_" + str(i) + ".gif"
        i = i + 1
        i1 = i1 + 1
        # 构造完整的文件路径
        file_path = os.path.join(".\\Images", image_name)
        
        # 将图片数据写入文件
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"图片已保存到 {file_path}")
        proccess_gif.process(file_path,f"{mode}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_" + str(i),mode)
        time.sleep(random.sample((0,2,0.1),1)[0])
        if i1 > 3:
            os.system('cls')
            i1 = 0
    else:
        i1 = 0
        print(f"图片下载失败，状态码：{response.status_code}")
        with open("log.txt","a+",encoding='utf-8') as log:
            log.write(f"at time {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} download file:{n_d} fail! response.status_code:{response.status_code} {response.text}")