#interesing_view.py
'''
*******************************************************************************************************
*******************************************************************************************************
**                                                                                                   **
**                                                                                                   **
**     IIIIII  NN   NN  TTTTTT   EEEEEEE   RRRRR    EEEEEE    SSSSS   IIIIII    OOOO    NN   NN      **
**       II    NNN  NN    TT    EE        RR   RR  EE        SS   SS    II    OO    OO  NNN  NN      **
**       II    NNNN NN    TT    EEEEEEEE  RR   RR  EEEEEEE    SS        II    OO    OO  NNNN NN      **
**       II    NN NNNN    TT    EEEEEEEE  RRRRR    EEEEEEE      SS      II    OO    OO  NN NNNN      **
**       II  　NN  NNN    TT    EE        RR  RR   EE        SS   SS    II    OO    OO  NN  NNN      **
**     IIIIII　NN   NN    TT     EEEEEEE  RR   RR   EEEEEE    SSSSS   IIIIII    OOOO    NN   NN      **
**                                                                                                   **
**                                                                                                   **
*******************************************************************************************************
*******************************************************************************************************
Interesion 是一款监控，脸部识别
interesing_view 是Interesion 的主循环模块

使用方法:
摄像头：
python interesing_model.py

视频：
python interesing_model.py -v [name*].[mp4|rmvb|flv|mpeg|avi|...]
'''
import cv2
import time
import pygame
import argparse
import numpy as np
from sys import exit
from interesing_model import exciting
# 创建参数解析器并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-n", "--name", type = str, default="Capture",help="window name")
ap.add_argument("-w", "--width", type = int,default=800,help="window width")
ap.add_argument("-ht", "--height", type = int,default=1000,help="window height")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")

args = vars(ap.parse_args())
# 如果video参数为None，那么我们从摄像头读取数据
if args.get("video", None) is None and args["image"] is None:
    camera = cv2.VideoCapture(0)
    #等待0.25秒
    time.sleep(0.25)
# 否则我们读取一个视频文件
else:
    camera = cv2.VideoCapture(args["video"])

width = args["width"]
height = args["height"]


et=exciting(camera,width=width,height=height)

while camera.isOpened():
	
	for event in pygame.event.get():
		if event.type == pygame.MOUSEBUTTONDOWN:
			pass
		if event.type == pygame.QUIT:
			camera.release()
			pygame.quit()
			exit()
	et.start
		