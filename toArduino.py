#toArduino.py
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
'''
import os
import cv2
import time
import serial
import pygame
from interesing_model import exciting

camera = cv2.VideoCapture(0)
et=exciting(camera)
val= 4 #初始的摄像头角度
r = 4 #脸部丢失时的回归角度
string='' #传递Arduino 字符串
time = 0 #识别到脸部次数
space = 6 #多少次脸部检查后的摄像头调整

w = 200
aw = 600

try:
	arduino = serial.Serial('com9',9600)
	print(arduino)
except:
	print('没链接到Arduino')

while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			camera.release()
			pygame.quit()
			exit()

		elif event.type == pygame.MOUSEMOTION:
		#return the X and Y position of the mouse cursor
			pos = pygame.mouse.get_pos()
			mouse_x = pos[0]
			mouse_y = pos[1]
		
	et.start
	try:
		x,y = et.toArduino()
		#print(x)
		
		if val >= 9:
			val = 9
		elif val <=0:
			val = 0

		if x != 0:
			time += 1
			if x < 280 and (time%space) == 0:
				val += 1
				i = str(val)
				arduino.write(i.encode())
				time.sleep(0.25)
			elif x > 470 and (time%space) == 0:
				val -= 1
				i = str(val)
				arduino.write(i.encode())
				time.sleep(0.25)
			
	except:
		pass
	'''

	x = et.toArduino()
	if x != None:
		if x < w:
			i = "0"
			print('左')
		elif x>aw:
			i = "1"
			print('右')
		arduino.write(i.encode())

	else:
		i = "404"
		arduino.write(i.encode())
'''
