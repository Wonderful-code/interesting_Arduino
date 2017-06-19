#pygame_model.py
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
Interesion 是一款监控,脸部识别
pygame_model 是Interesion 的pygame 显示模块

1.pygame主模块
2.显示模块

备忘录:
主色:
#F70022       #BB2D41       #A30016 	#FB415A		   #FB7487
(247,0,34)  (187,45,65)    (163,0,22)  (251,65,90)   (251,116,135)
辅助色 A:
#FF9500		  #C1852F		#A86300	    #FFB142	       #FFC676
(255,149,0)  (193,133,47)  (168,99,0)  (255,177,66)  (255,198,118)
辅助色 B:
#0068C5		  #246095		#004582		#3A93E3		   #69A9E3
(0,104,197)  (36,96,149)   (0,69,130)  (58,147,227)   (105,169,227)
互补色:
#37E600		  #4AAE2A		#259800		#6AF33E		   #90F370
(55,230,0)  (74,174,42)   (37,152,0)  (106,243,62)   (106,243,62)
'''
import cv2
import pygame
import datetime
import numpy as np
from math import pi
from sys import exit
from pygame.font import * 
from pygame.locals import *

class pygameDraw(object):
	def __init__(self,name,height,width):

		self._1 = 0
		self._name = name
		self._height = height
		self._width = width
		self.mouse_x = 0
		self.mouse_y = 0
		self._screen = None
				         #0          1
		self._color = [[0,0,0],[255,255,255],
						#新的脸2     3            4           5           6
					  [247,0,34],[187,45,65],[163,0,22],[251,65,90],[251,116,135],
					  #7             8            9          10           11
					  [255,149,0],[193,133,47],[168,99,0],[255,177,66],[255,198,118],
					  #12            13           14          15            16
					  [0,104,197],[36,96,149],[0,69,130],[58,147,227],[105,169,227],
					  #17            18          19           20         21
					  [55,230,0],[74,174,42],[37,152,0],[106,243,62],[106,243,62]]
		self.init

	@property
	def init(self):
		pygame.init()
		pygame.display.set_caption(self._name)
		self._screen = pygame.display.set_mode((self._width+200,self._height-399),pygame.RESIZABLE,32)
		self._screen.fill(self._color[0])#用黑色填充窗口

	def show_text(self, pos, text, color, font_bold = False, font_size = 13, font_italic = False):
		#Function:文字处理函数 pos:文字显示位置 color:文字颜色 font_bold:是否加粗 font_size:字体大小 font_italic:是否斜体 Output: NONE
		try:
		 	cur_font = pygame.font.SysFont("SourceHanSans-Bold", font_size)#字体，并设置文字大小 
		except:
			cur_font = pygame.font.SysFont("arial", font_size)#字体，并设置文字大小  
		cur_font.set_bold(font_bold)#设置是否加粗属性  
		cur_font.set_italic(font_italic)#设置是否斜体属性
		text_fmt = cur_font.render(text, 1, self._color[color])#设置文字内容 
		return self._screen.blit(text_fmt, pos)#绘制文字 

	def drawPeople(self,rect):
		for x,y,w,h in rect:
			pygame.draw.rect(self._screen,self._color[18],[x,y,w-x,h-y],3)
	def drawKNN(self,KNN):
		for r in KNN:
			x,y,w,h = self.wrap_digit(r)
			pygame.draw.rect(self._screen,self._color[17],[x,y,w,h],3)
	def drawFace(self,face,c=False):
		if c:
			color = 3 #新脸
		else:
			color = 4 #已识别
		for fx,fy,fw,fh in face:
			pygame.draw.rect(self._screen,self._color[color],[fx,fy,fw,fh],3)
	def drawConfirm(self,x,y,w,c=False):
		if c:
			color = 3 #新脸
		else:
			color = 4 #已识别
		pygame.draw.rect(self._screen,self._color[color],[x+w,y+(42*j),90,40])
	
	def drawFaces(self,faceShow=[]):
		if len(faceShow)<3:
			n = len(faceShow)
		else:
			n = 3
		for f in range(0,n):
			roj = np.swapaxes(faceShow[f],0, 1)
			roj = pygame.pixelcopy.make_surface(roj)
			self._screen.blit(roj,(self._width,f*200))
		'''
		for i in range(0,len(faceShow)+1):
			roj = np.swapaxes(faceShow[i],0, 1)
			roj = pygame.pixelcopy.make_surface(roj)
			#self.show_text(self._screen,(self._width-100,10),str(f),7,30)
			#self.show_text((self._width,i*200),str(i),6,40)
			self._screen.blit(roj,(self._width,i*200))
'''
	def Collision(self,x): #边界判断
		if (x-90) <= 0:
			return 0
		elif (x+90)>=self.width:
			return 1
			
	def inside(self,r1,r2):
		x1,y1,w1,h1 = r1
		x2,y2,w2,h2 = r2
		if (x1>x2) and (y1>y2) and (x1+w1<x2+w2) and (y1+h1<y2+h2):
			return True
		else:
			return False 

	def wrap_digit(self,rect):
		x,y,w,h = rect
		padding = 5
		hcenter = x+w/2
		vcenter = y+w/2
		if(h>w):
			w = h
			x = hcenter - (w/2)
		else:
			h = w
			y = vcenter - (w/2)
		return (x-padding,y-padding,w+padding,h+padding)

	def quit(self,camera):
		for event in pygame.event.get():
			if event.type == pygame.MOUSEBUTTONDOWN:
				pass
			if event.type == pygame.QUIT:
				camera.release()
				pygame.quit()
				exit()
			elif event.type == pygame.MOUSEMOTION:
			#return the X and Y position of the mouse cursor
				pos = pygame.mouse.get_pos()
				self.mouse_x = pos[0]
				self.mouse_y = pos[1]
'''
		视频: pygame.movie
要在游戏中播放片头动画、过场动画等视频画面，可以使用模块。

要播放视频中的音乐，pygame.movie模块需要对音频接口的完全控制，不能初始化mixer模块。因此要这样完成初始化

pygame.init()
pygame.mixer.quit()
或者只初始化 pygame.display.init()
movie = pygame.movie.Movie('filename') 指定文件名载入视频。视频的格式可以为mpeg1。视频文件不会马上全部载入内存，而是在播放的时候一点一点的载入内存。
movie.set_display(pygame.display.set_mode((640,480))) 指定播放的surface。
movie.set_volume(value) 指定播放的音量。音量的值value的取值范围为0.0到1.0。
movie.play() 播放视频。这个函数会立即返回，视频在后台播放。这个函数可以带一个参数loops，指定重复次数。 正在播放的视频可以用
movie.stop() 停止播放。
movie.pause() 暂停播放。
movie.skip(seconds) 使视频前进seconds秒钟。
NOTE：在某些XWindow下，可能需要配置环境变量： export SDL_VIDEO_YUV_HWACCEL=0
'''