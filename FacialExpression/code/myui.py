# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np
import math
import torch
from torchvision import transforms

########MyCode########
class FacialExpression():
    def __init__(self):
        self.emolist = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.model = torch.load(sys.path[0]+'/../model/vgg9net.pkl', map_location = torch.device('cpu'))
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def process_img(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            p = gray[y: y + h, x: x + w].copy()
            p = cv2.resize(p, (48, 48))
            p = torch.from_numpy(p).type(torch.FloatTensor).resize(1,1,48,48) #np->float32tensor
            t = self.model(p)
            _,emo = torch.max(t, 1) #概率最大的序号
            cv2.putText(image, self.emolist[emo], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        return image            
            
    def get_result(self, dir, data_type):
        imagelst = []
        fps = 0
        if data_type == 0: #pic
            image = cv2.imdecode(np.fromfile(dir, dtype = np.uint8),-1)
            image = self.process_img(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imagelst.append(image)
        else: #video
            cap=cv2.VideoCapture(dir)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                while True:
                    ret, frame = cap.read()  #read frame
                    if ret == False: #finish!
                        break
                    frame = self.process_img(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    imagelst.append(frame)
        return imagelst, fps

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1043, 721)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 580, 120, 80))
        self.groupBox.setObjectName("groupBox")
        self.picRadioButton = QtWidgets.QRadioButton(self.groupBox)
        self.picRadioButton.setGeometry(QtCore.QRect(0, 30, 115, 19))
        self.picRadioButton.setObjectName("picRadioButton")
        self.videoRadioButton = QtWidgets.QRadioButton(self.groupBox)
        self.videoRadioButton.setGeometry(QtCore.QRect(0, 60, 115, 19))
        self.videoRadioButton.setObjectName("videoRadioButton")
        self.selectPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.selectPushButton.setGeometry(QtCore.QRect(150, 590, 93, 28))
        self.selectPushButton.setObjectName("selectPushButton")
        self.playPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.playPushButton.setGeometry(QtCore.QRect(150, 620, 93, 28))
        self.playPushButton.setObjectName("playPushButton")
        self.Slider = QtWidgets.QSlider(self.centralwidget)
        self.Slider.setGeometry(QtCore.QRect(260, 600, 761, 31))
        self.Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Slider.setObjectName("Slider")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(0, 10, 1041, 541))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1039, 539))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.monitorLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.monitorLabel.setEnabled(True)
        self.monitorLabel.setGeometry(QtCore.QRect(10, 10, 1021, 521))
        self.monitorLabel.setObjectName("monitorLabel")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1043, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        ########MyCode########
        #init ui
        self.selectPushButton.setEnabled(True)
        self.playPushButton.setEnabled(False)
        self.Slider.setEnabled(False)
        self.picRadioButton.setChecked(True)        
        #timer
        self.timer = QtCore.QTimer() #init timer
        self.timer.timeout.connect(self.play_video) #call func when timeout
        #signal-slot
        self.selectPushButton.clicked.connect(self.OnClickedBtnSel)
        self.playPushButton.clicked.connect(self.OnClickedBtnPlay)
        self.Slider.valueChanged.connect(self.OnSliderChanged)
        #member
        self.image = None
        self.fps = 0
        self.frame_len = 0
        self.frame_num = 0
        self.play_mode = 0

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "FileType"))
        self.picRadioButton.setText(_translate("MainWindow", "Picture"))
        self.videoRadioButton.setText(_translate("MainWindow", "Video"))
        self.selectPushButton.setText(_translate("MainWindow", "SelectFile"))
        self.playPushButton.setText(_translate("MainWindow", "Play/Stop"))
        self.monitorLabel.setText(_translate("MainWindow", "FacialExpressionRecognition"))

    ########MyCode########
    def OnClickedBtnSel(self):
        if self.picRadioButton.isChecked() == True:
            dir = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "Select", "", "Image (*.png *.jpg *.jpeg *.bmp)")[0]
            self.data_type = 0 #pic
            self.playPushButton.setEnabled(False)
            self.Slider.setEnabled(False)
        else:
            dir = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "Select", "", "Video (*.avi *.mpeg *.mpg *.mp4 *.rmvb *.flv)")[0]
            self.data_type = 1 #video
            self.playPushButton.setEnabled(True)
            self.Slider.setEnabled(True)

        self.image = None
        self.fps = 0
        self.frame_len = 0
        self.frame_num = 0
        self.play_mode = 0        
        self.Slider.disconnect()
        self.Slider.setValue(0) #Slider先断开槽函数，置0，再连接
        self.Slider.valueChanged.connect(self.OnSliderChanged)

        f = FacialExpression()
        self.image, self.fps = f.get_result(dir, self.data_type)
        self.frame_len = len(self.image)
        if self.data_type == 1:
            print('fps = %f, frame = %d' % (self.fps, self.frame_len))
            self.timer.setInterval(int(1000/(self.fps)))
        self.show()

    def show(self, frame_num = 0):
        h, w = self.monitorLabel.height(), self.monitorLabel.width()
        image = self.image[frame_num]
        k = min(h / image.shape[0], w / image.shape[1])
        h1 = math.floor(k * image.shape[0])
        w1 = math.floor(k * image.shape[1])
        
        self.scrollAreaWidgetContents.setMinimumSize(image.shape[1], image.shape[0])
        self.monitorLabel.resize(image.shape[1], image.shape[0])
        image = QtGui.QImage(image[:],image.shape[1], image.shape[0],image.shape[1] * 3, QtGui.QImage.Format_RGB888)
        
        pixmap = QtGui.QPixmap.fromImage(image)
        self.monitorLabel.setPixmap(pixmap)

    def play_video(self):
        self.show(self.frame_num)
        self.frame_num = min(self.frame_num + 1, self.frame_len - 1)
        self.Slider.disconnect()
        self.Slider.setValue(100 * self.frame_num/(self.frame_len - 1))
        self.Slider.valueChanged.connect(self.OnSliderChanged)
        if self.frame_num >= self.frame_len - 1:
            self.timer.stop()

    def OnClickedBtnPlay(self):
        if self.play_mode == 0:
            self.play_mode = 1
            self.timer.start()
        else:
            self.play_mode = 0
            self.timer.stop()

    def OnSliderChanged(self):
        self.frame_num = int((self.frame_len - 1) * self.Slider.value()/100)
        self.show(self.frame_num) 

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(w)
    w.show()
    sys.exit(app.exec_())