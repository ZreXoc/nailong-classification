import time
import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# 打开GIF文件
n_frames = 4  # 想要抽取的帧数

def process(gif_path,suf,mode=0,output_folder = ''):
    reader = imageio.get_reader(gif_path)
    if output_folder == '':
        if mode == 0:
            output_folder = "D:\\nailong-classification\\data\\train\\bqb"
        elif mode == 1:
            output_folder = "D:\\nailong-classification\\data\\train\\nailong"
        elif mode == 2:
            output_folder = "D:\\nailong-classification\\data\\test\\bqb"
        elif mode == 3:
            output_folder = "D:\\nailong-classification\\data\\test\\nailong"
    total_frames = reader.get_length()
    random.seed(suf+str(total_frames)+str(time.time()))
    # 随机选择帧
    frames_to_extract = random.sample(range(total_frames), n_frames if n_frames < total_frames else total_frames)
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file = ''
    filename = ''
    filetr = []
    # 抽取帧并保存
    for i, frame_number in enumerate(frames_to_extract):
        reader.set_image_index(frame_number)
        frame = reader.get_data(frame_number)
        filename = f"{output_folder}/{suf}_frame_{frame_number}_pic_{i}.png"
        file += filename + ' '
        filetr.append(filename)
        imageio.imwrite(filename, frame)
    
    print(f"抽帧完成 文件：{file}")
    reader.close()
    return filetr,n_frames
# print(f"Extracted frames {frames_to_extract} and saved as {[frame_path for i, frame_number in enumerate(frames_to_extract)]}")