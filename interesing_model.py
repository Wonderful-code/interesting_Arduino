#interesion_model.py
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
interesion_model 是Interesion 的核心模块

1.进行身份识别
2.记录脸部图片 以200*200大小保存图片

备忘录：　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
面人脸分类器进行了实验，总共有4个，alt、alt2、alt_tree、default。
对比下来发现alt和alt2的效果比较好，alt_tree耗时较长，default是一个轻量级的，
经常出现误检测。所以还是推荐大家使用haarcascade_frontalface_atl.xml和
haarcascade_frontalface_atl2.xml。

'''
import os
import sys
import cv2
import dlib
import time
import json
import pygame
import imutils
import datetime
import numpy as np 
from pygame_model import pygameDraw
from imutils.object_detection import non_max_suppression

class Id(object):

	def __init__(self,Id):
		self.face=[]
		self._ID = Id
		self._faceN = -1
		self._grayfile = 'face/face_gray/'
		self._colorfile='face/face_color/'
		self.faceN()
	def setId(self,Id):
		self._ID = Id
	@property
	def id(self):
		return self._ID
	#当前脸图片数量
	def faceN(self):
		try:
			for i in range(0,22):
				self.face.append(cv2.imread(self._colorfile+str(self._ID)+'/%s.png' % str(i),cv2.IMREAD_COLOR))
				self._faceN = i
		except:
			pass
	#保存脸部图片
	def face(self,color,gray):
		
		if self._faceN <=20:
			try:
				os.mkdir(self._grayfile + str(self._ID))
				os.mkdir(self._colorfile + str(self._ID))
			except Exception as e:
				pass
			path_gray = self._grayfile + str(self._ID) +'/'+ str(self._faceN) + '.png'
			path_color = self._colorfile + str(self._ID) +'/'+ str(self._faceN) + '.png'
			cv2.imwrite(path_gray,gray)
			cv2.imwrite(path_color,color)

class exciting(object):

	def __init__(self,camera,#摄像头对象
					name='Interesing',
					width =800,
					height=1000,
					history=20,#背景建模样本量
					args=500):#忽略大小
					
		self.args = args
		self.name = name
		self.width = width
		self.height = height
		self.camera = camera
		self.history = history

		self.time = None
		self.gray = None
		self.model = None
		self._frame = None
		self._color = None
		self._params = None
		self._startTime = None
		self._backgrouds = None
		self._fpsEstimate = None
		self._videoWriter = None
		self._videoFilename = None
		self._videoEncoding = None
		self._facearray=None
		
		self._frames = 0 #帧计数器
		self._firstFace = 0
		self._framesElapsed = 0

		self._faceShow=[]
		self._face_IDs=[]
		self._data = {}

		self.detector = dlib.get_frontal_face_detector()
		
		self.hog = cv2.HOGDescriptor()#初始化方向梯度直方图描述子/设置支持向量机
		self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		self.bg = cv2.createBackgroundSubtractorKNN(detectShadows=True)#初始化背景分割器
		self.bg.setHistory(self.history)
		
		self.face_alt2 = cv2.CascadeClassifier('haarcascades//haarcascade_frontalface_alt2.xml')
		self.path = ['Background','face','face/face_gray','face/face_color']

		self.imgflie
		self.facesID
		self.pd = pygameDraw(self.name,self.height,self.width)
	@property
	def facesID(self):
		try:
			with open('face.json', 'r',encoding='UTF-8') as f:
				self._data=json.load(f)
				for i in range(0,len(self._data.keys())+1):
					self._face_IDs.append(Id(i))
		except:
			pass
		
	def setface(self,face):
		with open('face.json','w', encoding='utf8') as f:
			json.dump(self._data,f,ensure_ascii=False)
	
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

	def Collision(self,x):
		if (x-90) <= 0:
			return True
		elif (x+90)>=self.width:
			return False
		
	@property
	def imgflie(self):
		for path in self.path:
			if os.path.exists(path):
				print("OK")
			else:
				os.mkdir(path)
	@property #在写视频吗？
	def isWritingVideo(self):
		return self._videoFilename is not None
	@property
	def FPS(self):
		#更新FPS估计和相关变量。
		if self._framesElapsed == 0:
			self._startTime = time.time()
		else:
			timeElapsed = time.time() - self._startTime
			self._fpsEstimate = self._framesElapsed/timeElapsed
		self._framesElapsed += 1
	@property
	def backgrouds():
		self._backgrouds = self.readBackgroud(random.randint(0,19))
		if self._backgrouds != None:
			self._frames = 20
	def readBackgroud(self,count):
		return cv2.imread('Background//%s.png' % str(count),cv2.IMREAD_COLOR)

	def writeBackgroud(self,count):#(帧，张)
		return cv2.imwrite('Background//%s.png' % str(count),self._frame)

	def writeImg(self,frame,path,count):
		return cv2.imwrite(path+'/%s.png' % str(count),frame)

	def read_images_array(self,path='face/face_gray/',l=20,z=0):
	#读取图片（路径，张数，人数，图片太大了）
	#face/str(1)/str(i)+'png'
		c=0
		x,y=[],[]
		for o in range(0,z+1):
			for i in range(0,l+1):
				try:
					path_in = path +str(o+1) +'/'+ str(i) + '.png'
					im = cv2.imread(path_in,cv2.IMREAD_GRAYSCALE)
					im = imutils.resize(im,32,32)
				#im = cv2.resize(im,(100,100),interpolation = cv2.INTER_LINEAR)
				
					x.append(np.asarray(im,dtype=np.uint8))
					y.append(c)
					path_in = None
				except:
					continue
			c=c+1
		self._facearray = [x,y]
	@property
	def face_rec(self):
		if self._facearray:
			[x,y] = self._facearray
			y=np.asarray(y,dtype=np.int32)
			self.model = cv2.face.createEigenFaceRecognizer()
			self.model.train(np.asarray(x),np.asarray(y))
	def _writeVideoFrame(self):
		if not self.isWritingVideo:
			return
		if self._videoWriter is None:
			fps = self.camera.get(cv2.CAP_PROP_FPS)
			if fps == 0.0:
				if self._framesElapsed <20:
					return
				else:
					fps = self._fpsEstimate
			size = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
			self._videoWriter = cv2.VideoWriter(self._videoFilename,self._videoEncoding,fps,size)
		self._videoWriter.Write(self._frame)
	#开始录像
	def Monitor(self,path,frame,encoding = cv2.VideoWriter_fourcc('I','4','2','0')):
		self._videoFilename = path
		self._videoEncoding = encoding
		self._writeVideoFrame()

	def confirm(self,face,faces):#face 区域，faces 身份
		for i in range(0,len(faces)):
			for j in range(0,len(faces[i])):
				Collision=self.Collision(x)
				self.pd.drawConfirm(face)
				self.pd.show_text(self.pd._screen,(fx+fw,fy+(45*j)),faces[i][j],2, True,30)
	#基础脸部识别		
	def face(self,gray):
		return self.face_alt2.detectMultiScale(gray, 1.3, 5)#脸
	#身份识别
	def face2(self,roi):
		try:
			return self.model.predict(roi)
		except:
			return
	#dlib 脸识别器
	def dlibFace(self,frame):
		l = None
		dets = self.detector(frame, 0)
		for i,d in enumerate(dets):
			#pygame.draw.rect(screen,[163,0,22],[d.left(),d.top(),d.right(),d.bottom()],3)
			return d.left(),d.top(),d.right(),d.bottom()
	#两帧不同
	def frame_difference(self,firstFrame,gray):
		avg = cv2.cvtColor(firstFrame,cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray,(21,21),0)
		avg = cv2.GaussianBlur(avg,(21,21),0)
		#cv2.accumulateWeighted(gray,avg,0.5)
		frameDelta =cv2.absdiff(gray, cv2.convertScaleAbs(avg))
		thresh =cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
		# 扩展阀值图像填充孔洞，然后找到阀值图像上的轮廓
		thresh =cv2.dilate(thresh, None, iterations=2)
		(_,cnts, _) =cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		return cnts
	#KNN
	def KNN_difference(self,frame,min_area=500):
		fgmask = self.bg.apply(frame)
		thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)
		#findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy
		(_,cnts,_)= cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		rectangles = []
		for c in cnts:
			if cv2.contourArea(c) < min_area:
				continue
			r = (x, y, w, h) = cv2.boundingRect(c)

			is_inside = False
			for q in rectangles:
				if self.inside(r,q):
					is_inside = True
					break		
			if not is_inside:
				rectangles.append(r)
		return rectangles
	#人识别
	def people(self,frame):
		#调整到（1）减少检测时间，将图像裁剪到最大宽度为400个像素
		frame = imutils.resize(frame, width=min(400, frame.shape[1]))
		#检测图像中的人
		(rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4),
			padding=(8, 8), scale=1.05)
		# 应用非极大值抑制的边界框
		# 相当大的重叠阈值尽量保持重叠
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		return non_max_suppression(rects, probs=None, overlapThresh=0.65)
	#背景建模
	@property
	def bgbuild(self):
		try:
			back = self.readBackgroud(5)
		except:
			if self._frames < self.history:
				
				self.pd.show_text((100,200),u"请离开镜头",1,True,120)
				self.pd.show_text((110,320),u"背      景      建      模      中:  {}%".format((self._frames/self.history)*100),1,True,40)

				KNN=self.KNN_difference(self._frame,self.args)
				rect = self.people(self._frame) #人检查
				l,t,w,h= self.dlibFace(self._color)
				face = self.face(self.gray) #脸
			
				if rect != [] or face != () or l != None or KNN != []:
					self.pd.drawKNN(KNN)
					self.pd.drawPeople(rect)
					self.pd.drawFace(face)
				else:
					self.writeBackgroud(self._frames)
					self._frames += 1

	@property
	def start(self):
		self.pd.quit(self.camera)
		self.FPS
		(ret,cv_img)= self.camera.read()
		self._frame = imutils.resize(cv_img,self.width,self.height)
		self.gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)#灰色
		self._color = cv2.cvtColor(self._frame, cv2.COLOR_RGB2BGR)#opencv的色彩空间是BGR，pygame的色彩空间是RGB
		try:
			self.pd.show_text((10,self._frame.shape[0]-40),datetime.datetime.now().strftime(u"%Y-%d %I:%M:%S"),15,True,30)
			pygame.display.update()
			pixl_arr = np.swapaxes(self._color, 0, 1)
			new_surf = pygame.pixelcopy.make_surface(pixl_arr)
			self.pd._screen.blit(new_surf, (0, 0))
		except:
			pass
		self.discern()
		if self._faceShow != []:
			self.pd.drawFaces(self._faceShow)
		self.bgbuild#背景建模

	def discern(self):
		KNN=self.KNN_difference(self._frame,self.args)
		if  KNN != []:
			face=self.face(self.gray) #脸
			if face != ():
				try:
					self.pd.drawFace(face)#显示区域
				except:
					pass
				x,y = [],[]
				for fx,fy,fw,fh in face:
					roi = self.gray[fy:fy+fh,fx:fx+fw]
					roj = self._color[fy:fy+fh,fx:fx+fw]
					roi = cv2.resize(roi,(200,200))
					roj = cv2.resize(roj,(200,200))
					self._faceShow.append(roj)

					if self._facearray != None:
						self.face_rec
						self._params = self.face2(roi)

				if self._params != None:
					self.pd.drawFace(face,True)
					print(self._params[0],self._params[1])
					print("0: %s, Confidence: %.2f" % (self._params[0],self._params[1]))
					#n = self._face_IDs[self._params[0]].faceN()
					#self._face_IDs.append(Id(len(self._face_IDs)))

	def toArduino(self):
		l = 0
		t = 0
		KNN=self.KNN_difference(self._frame,self.args)
		if  KNN != []:
			face=self.face(self.gray) #脸

			try:
				l,t,w,h = self.dlibFace(self._color)
			except:
				pass
			if face != ():
				self.pd.drawFace(face)#显示区域
			return l,t
''' 
#人脸标记 #匹配值
self._facearray=self.read_images_array()

roi = cv2.resize(roi,(32,32),interpolation = cv2.INTER_LINEAR)
					params[0],params[1]
					self._data
					x.append(np.asarray(roi,dtype=np.uint8))
					y.append(faceslen)
					et.face_rec([x,y])
					cv2.imwrite('face/face_gray/1/%s.png' % str(faceID),roi)
					cv2.imwrite('face/face_color/1/%s.png' % str(faceID),roj)

					faceslen = faceslen+1
				else:
					et.face2(roj)
					if f<20:
					cv2.imwrite('face/face_gray/1/%s.png' % str(f),roi)
					cv2.imwrite('face/face_color/1/%s.png' % str(f),roj)
					f =f+1
'''
	

def car(self):
	#汽车
	pass

def Ev(self):
	#电动车
	pass

def bicycle(self):
	#自行车
	pass