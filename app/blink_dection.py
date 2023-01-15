# -*- coding: utf-8 -*-
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np # 数据处理的库 numpy
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])# 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear


# PATH_PREDICTOR = "/home/pi/dev/shape_predictor_68_face_landmarks.dat"
PATH_PREDICTOR = "app/models/shape_predictor_68_face_landmarks.dat"

class BlinkDetection():
    def __init__(self,EYE_AR_THRESH=0.2,EYE_AR_CONSEC_FRAMES=3):
        super().__init__()

        print("[INFO] loading facial landmark predictor...")

        self.EYE_AR_THRESH = EYE_AR_THRESH
        self.EYE_AR_CONSEC_FRAMES = EYE_AR_CONSEC_FRAMES
        self.COUNTER = 0
        self.TOTAL = 0

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PATH_PREDICTOR)

        self.lStart = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]
        self.lEnd  = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]
        self.rStart = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]
        self.rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]

    def reset(self):
        self.TOTAL = 0
        self.COUNTER = 0



    def detection(self, frame):
        frame = imutils.resize(frame, width=720)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = self.detector(gray, 0)
        # 脸部未检测到，rects的长度为0，此时存在注意力不集中的潜在可能性
        # if len(rects) is 0:
        #     print("potential no-attention!")

        # 第七步：循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
        for rect in rects:
            shape = self.predictor(gray, rect)

            # 第八步：将脸部特征信息转换为数组array的格式
            shape = face_utils.shape_to_np(shape)

            # 第九步：提取左眼和右眼坐标
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]

            # 第十步：构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            self.ear = (leftEAR + rightEAR) / 2.0

            # 第十一步：使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
            # leftEyeHull = cv2.convexHull(leftEye)
            # rightEyeHull = cv2.convexHull(rightEye)
            # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # 第十二步：进行画图操作，用矩形框标注人脸
            # left = rect.left()
            # top = rect.top()
            # right = rect.right()
            # bottom = rect.bottom()
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

            '''
                分别计算左眼和右眼的评分求平均作为最终的评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示进行了一次眨眼活动
            '''
            # 第十三步：循环，满足条件的，眨眼次数+1
            if self.ear < self.EYE_AR_THRESH:  # 眼睛长宽比：0.2
                self.COUNTER += 1

            else:
                # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:  # 阈值：3
                    self.TOTAL += 1
                # 重置眼帧计数器
                self.COUNTER = 0

            #seconds = time.time() - CURRENT_TIME
            #print("Frame Time take : {0}seconds".format(seconds))
            #fps = NUM_FRAMES / (time.time() - START_TIME)
            #print("realtime_fps : ", fps)

            # 第十四步：进行画图操作，68个特征点标识
            # for (x, y) in shape:
            #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # 第十五步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示
            # cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame, "Blinks: {}".format(self.TOTAL), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame, "COUNTER: {}".format(self.COUNTER), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame, "EAR: {:.2f}".format(self.ear), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # print('眼睛实时长宽比:{:.2f} '.format(self.ear))
        # if self.TOTAL >= 50:
        #     cv2.putText(frame, "SLEEP!!!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # cv2.putText(frame, "Press 'q': Quit", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
        # 窗口显示 show with opencv

        # cv2.imshow("Frame", frame)
        # with lock:
        #     outframe = frame.copy()
        # with lock:
        #     flag, encodeImage = cv2.imencode(".jpg",outframe)
        # yield ('--frame\r\n'  'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodeImage) +'\r\n')

        return self.TOTAL
