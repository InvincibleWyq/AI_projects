# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'llk.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets
import sys, os, math, timeit
from llkai import *

# get pic directory
picdir = './pic/'
picname = os.listdir(picdir)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1237, 865)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.labelChessBoard = QtWidgets.QLabel(self.centralwidget)
        self.labelChessBoard.setGeometry(QtCore.QRect(530, 20, 681, 681))
        self.labelChessBoard.setObjectName("labelChessBoard")
        self.labelUsage = QtWidgets.QLabel(self.centralwidget)
        self.labelUsage.setGeometry(QtCore.QRect(30, 710, 1181, 91))
        self.labelUsage.setObjectName("labelUsage")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(20, 390, 211, 216))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.textEdit = QtWidgets.QTextEdit(self.widget)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout.addWidget(self.textEdit)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(20, 260, 211, 111))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.groupBox = QtWidgets.QGroupBox(self.widget1)
        self.groupBox.setObjectName("groupBox")
        self.radioButtonAuto = QtWidgets.QRadioButton(self.groupBox)
        self.radioButtonAuto.setGeometry(QtCore.QRect(10, 20, 91, 19))
        self.radioButtonAuto.setObjectName("radioButtonAuto")
        self.radioButtonManual = QtWidgets.QRadioButton(self.groupBox)
        self.radioButtonManual.setGeometry(QtCore.QRect(10, 50, 91, 19))
        self.radioButtonManual.setObjectName("radioButtonManual")
        self.horizontalLayout_6.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.widget1)
        self.groupBox_2.setObjectName("groupBox_2")
        self.radioButtonMode1 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButtonMode1.setGeometry(QtCore.QRect(0, 20, 71, 19))
        self.radioButtonMode1.setObjectName("radioButtonMode1")
        self.radioButtonMode2 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButtonMode2.setGeometry(QtCore.QRect(0, 40, 71, 19))
        self.radioButtonMode2.setObjectName("radioButtonMode2")
        self.radioButtonMode3 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButtonMode3.setGeometry(QtCore.QRect(0, 60, 71, 19))
        self.radioButtonMode3.setObjectName("radioButtonMode3")
        self.horizontalLayout_6.addWidget(self.groupBox_2)
        self.widget2 = QtWidgets.QWidget(self.centralwidget)
        self.widget2.setGeometry(QtCore.QRect(20, 630, 211, 69))
        self.widget2.setObjectName("widget2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.pushButtonGenerate = QtWidgets.QPushButton(self.widget2)
        self.pushButtonGenerate.setObjectName("pushButtonGenerate")
        self.horizontalLayout_7.addWidget(self.pushButtonGenerate)
        self.pushButtonSolve = QtWidgets.QPushButton(self.widget2)
        self.pushButtonSolve.setObjectName("pushButtonSolve")
        self.horizontalLayout_7.addWidget(self.pushButtonSolve)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.pushButtonBack = QtWidgets.QPushButton(self.widget2)
        self.pushButtonBack.setObjectName("pushButtonBack")
        self.horizontalLayout_8.addWidget(self.pushButtonBack)
        self.pushButtonFoward = QtWidgets.QPushButton(self.widget2)
        self.pushButtonFoward.setObjectName("pushButtonFoward")
        self.horizontalLayout_8.addWidget(self.pushButtonFoward)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.widget3 = QtWidgets.QWidget(self.centralwidget)
        self.widget3.setGeometry(QtCore.QRect(20, 20, 211, 221))
        self.widget3.setObjectName("widget3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget3)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEditM = QtWidgets.QLineEdit(self.widget3)
        self.lineEditM.setObjectName("lineEditM")
        self.horizontalLayout.addWidget(self.lineEditM)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget3)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEditN = QtWidgets.QLineEdit(self.widget3)
        self.lineEditN.setObjectName("lineEditN")
        self.horizontalLayout_2.addWidget(self.lineEditN)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.widget3)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.lineEditK = QtWidgets.QLineEdit(self.widget3)
        self.lineEditK.setObjectName("lineEditK")
        self.horizontalLayout_3.addWidget(self.lineEditK)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.widget3)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.lineEditP = QtWidgets.QLineEdit(self.widget3)
        self.lineEditP.setObjectName("lineEditP")
        self.horizontalLayout_4.addWidget(self.lineEditP)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.widget3)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        self.lineEditZ = QtWidgets.QLineEdit(self.widget3)
        self.lineEditZ.setObjectName("lineEditZ")
        self.horizontalLayout_5.addWidget(self.lineEditZ)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.widget4 = QtWidgets.QWidget(self.centralwidget)
        self.widget4.setGeometry(QtCore.QRect(250, 20, 251, 111))
        self.widget4.setObjectName("widget4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget4)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_8 = QtWidgets.QLabel(self.widget4)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_9.addWidget(self.label_8)
        self.lcdNumberG = QtWidgets.QLCDNumber(self.widget4)
        self.lcdNumberG.setObjectName("lcdNumberG")
        self.horizontalLayout_9.addWidget(self.lcdNumberG)
        self.verticalLayout_4.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_9 = QtWidgets.QLabel(self.widget4)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_10.addWidget(self.label_9)
        self.lcdNumberH = QtWidgets.QLCDNumber(self.widget4)
        self.lcdNumberH.setObjectName("lcdNumberH")
        self.horizontalLayout_10.addWidget(self.lcdNumberH)
        self.verticalLayout_4.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_10 = QtWidgets.QLabel(self.widget4)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_11.addWidget(self.label_10)
        self.lcdNumberF = QtWidgets.QLCDNumber(self.widget4)
        self.lcdNumberF.setObjectName("lcdNumberF")
        self.horizontalLayout_11.addWidget(self.lcdNumberF)
        self.verticalLayout_4.addLayout(self.horizontalLayout_11)
        self.widget5 = QtWidgets.QWidget(self.centralwidget)
        self.widget5.setGeometry(QtCore.QRect(250, 150, 258, 551))
        self.widget5.setObjectName("widget5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget5)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_11 = QtWidgets.QLabel(self.widget5)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_5.addWidget(self.label_11)
        self.listWidget = QtWidgets.QListWidget(self.widget5)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_5.addWidget(self.listWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1237, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        #my own variable
        self.label_list = [] #显示图像
        self.lst = [] #用来存结果，包含每一步的棋盘
        self.cursor = 0 #表示正在查看第几步
        self.turn_count = 0 #g
        self.path_cost = 0 #f

        #initialize
        self.radioButtonAuto.setChecked(True)
        self.radioButtonMode1.setChecked(True)
        self.lineEditM.setText('4')
        self.lineEditN.setText('4')
        self.lineEditK.setText('7')
        self.lineEditP.setText('4')
        self.lineEditZ.setText('0')
        self.pushButtonGenerate.setEnabled(True)
        self.pushButtonSolve.setEnabled(False)
        self.pushButtonBack.setEnabled(False)
        self.pushButtonFoward.setEnabled(False)

        #connect signal and slotfunction
        self.pushButtonGenerate.clicked.connect(self.OnClickedGenerate)
        self.pushButtonSolve.clicked.connect(self.OnClickedSolve)
        self.pushButtonBack.clicked.connect(self.OnClickedBack)
        self.pushButtonFoward.clicked.connect(self.OnClickedFoward)
    
    def set_one_label(self, i, j):
        p = i*self.problem.n + j
        if self.problem.display_state[i+1,j+1] == -1:
            pic = QtGui.QPixmap(picdir + '0.png').scaled(self.label_list[p].width(), self.label_list[p].height())
            self.label_list[p].setPixmap(pic)
        elif self.problem.display_state[i+1,j+1] == 0:
            self.label_list[p].setStyleSheet("background-color: rgb(0, 0, 0);")
        else:
            pic = QtGui.QPixmap(picdir + picname[int(self.problem.display_state[i+1,j+1])])\
                .scaled(self.label_list[p].width(), self.label_list[p].height())
            self.label_list[p].setPixmap(pic)
            
    def display(self):
        [h, w] = [self.labelChessBoard.height(), self.labelChessBoard.width()]        
        labelh = math.floor((h - self.problem.m - 1)/self.problem.m)
        labelw = math.floor((w - self.problem.n - 1)/self.problem.n)
        labelmin = min(labelw, labelh)
        startx = math.floor((w - (labelmin + 1) * self.problem.n + 1)/2)
        starty = math.floor((h - (labelmin + 1) * self.problem.m + 1)/2)        
        while self.label_list != []:
            l = self.label_list.pop()
            l.hide()
        for i in range(self.problem.m):
            for j in range(self.problem.n):
                p = i*self.problem.n + j
                self.label_list.append(QtWidgets.QLabel(self.labelChessBoard))
                self.label_list[p].setGeometry(QtCore.QRect(\
                    startx+j*(labelmin+1), starty+i*(labelmin+1), labelmin, labelmin))
                self.label_list[p].setObjectName("board_label_" + str(i) + '_' + str(j))
                self.set_one_label(i, j)
                self.label_list[p].show()

    def OnClickedGenerate(self):        
        if self.radioButtonMode1.isChecked():
            self.mode = 1
        elif self.radioButtonMode2.isChecked():
            self.mode = 2
        elif self.radioButtonMode3.isChecked():
            self.mode = 3

        if self.radioButtonAuto.isChecked():
            auto = 1
        elif self.radioButtonManual.isChecked():
            auto = 0

        self.m = int(self.lineEditM.text())
        self.n = int(self.lineEditN.text())
        self.k = int(self.lineEditK.text())
        self.p = int(self.lineEditP.text())        
        self.z = 0 if (self.mode!=3) else int(self.lineEditZ.text())

        #检查自动生成的参数的合法性
        valid = (2*self.k+self.z)<=(self.m*self.n) and (self.k>=self.p)
        valid = valid or (auto==0)
        if not valid: 
            QtWidgets.QMessageBox.information(self.centralwidget, "警告","参数不合法！")
            return
        
        self.problem = LLKProblem(self.mode, self.m, self.n, self.k, self.p, self.z)     

        if auto==0:#手动对init_node赋值
            matrix = self.textEdit.toPlainText()
            rows = matrix.split('\n')
            for i in range(self.m):
                cols = rows[i].split(' ')
                for j in range(self.n):
                    self.problem.init_state[i+1][j+1] = cols[j]
            self.problem.init_node = Node(self.problem.init_state)

        print("\n连连看棋盘初始状态：")
        print(self.problem.init_state)
        self.display()
        self.listWidget.clear()
        self.lcdNumberG.display(self.turn_count)
        self.lcdNumberH.display(self.path_cost-self.turn_count)
        self.lcdNumberF.display(self.path_cost)
        self.pushButtonGenerate.setEnabled(True)
        self.pushButtonSolve.setEnabled(True)
        self.pushButtonBack.setEnabled(False)
        self.pushButtonFoward.setEnabled(False)

    def OnClickedSolve(self):
        self.cursor = 0
        start=timeit.default_timer()
        self.lst = search(self.problem)
        end=timeit.default_timer()
        self.pushButtonGenerate.setEnabled(True)
        self.pushButtonSolve.setEnabled(False)
        self.pushButtonBack.setEnabled(True)
        self.pushButtonFoward.setEnabled(True)
        self.OnClickedFoward()
        self.OnClickedBack()
        QtWidgets.QMessageBox.information(self.centralwidget, "提示",\
            "使用A*求解连连看，最大分支%d，最多拐%d次\n运行时间: %.5s 秒"\
                %(self.problem.maxbranch, self.problem.maxturn,end-start))

    def OnClickedBack(self):
        if self.cursor > 0:
            self.cursor -= 1
            self.problem.display_state = self.lst[self.cursor].state
            self.display()
            self.turn_count = self.lst[self.cursor].turn_count
            self.path_cost = self.lst[self.cursor].path_cost
            self.lcdNumberG.display(self.turn_count)
            self.lcdNumberH.display(self.path_cost-self.turn_count)
            self.lcdNumberF.display(self.path_cost)
            self.listWidget.takeItem(self.listWidget.count()-1)
            self.listWidget.setCurrentRow(self.listWidget.count()-1)

    def OnClickedFoward(self):
        if self.cursor < len(self.lst) - 1:
            self.cursor += 1
            self.problem.display_state = self.lst[self.cursor].state
            self.display()
            self.turn_count = self.lst[self.cursor].turn_count            
            self.path_cost = self.lst[self.cursor].path_cost
            self.lcdNumberG.display(self.turn_count)
            self.lcdNumberH.display(self.path_cost-self.turn_count)
            self.lcdNumberF.display(self.path_cost)
            self.listWidget.addItem('turn_count:'+str(self.turn_count)+' path_cost:'+str(self.path_cost))
            self.listWidget.setCurrentRow(self.listWidget.count()-1)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "连连看"))
        self.labelUsage.setText(_translate("MainWindow", "模式1只允许两次以内的转向，模式2允许两次以上转向，模式3进一步允许出现阻断格子。即使手动输入，也要保证mnkpz的合法性！"))
        self.label_6.setText(_translate("MainWindow", "手动输入初始棋盘"))
        self.textEdit.setPlaceholderText(_translate("MainWindow", "请输入m行n列矩阵，每行元素间隔一个空格，0表示空元素，1~p之间的数值表示某类图像元素，负值表示阻断元素"))
        self.groupBox.setTitle(_translate("MainWindow", "输入方式"))
        self.radioButtonAuto.setText(_translate("MainWindow", "自动生成"))
        self.radioButtonManual.setText(_translate("MainWindow", "手动输入"))
        self.groupBox_2.setTitle(_translate("MainWindow", "模式选择"))
        self.radioButtonMode1.setText(_translate("MainWindow", "模式1"))
        self.radioButtonMode2.setText(_translate("MainWindow", "模式2"))
        self.radioButtonMode3.setText(_translate("MainWindow", "模式3"))
        self.pushButtonGenerate.setText(_translate("MainWindow", "生成棋盘"))
        self.pushButtonSolve.setText(_translate("MainWindow", "开始求解"))
        self.pushButtonBack.setText(_translate("MainWindow", "上一步"))
        self.pushButtonFoward.setText(_translate("MainWindow", "下一步"))
        self.label.setText(_translate("MainWindow", "棋盘行数 m"))
        self.label_2.setText(_translate("MainWindow", "棋盘列数 n"))
        self.label_3.setText(_translate("MainWindow", "图案对数 k"))
        self.label_4.setText(_translate("MainWindow", "图案类别 p"))
        self.label_5.setText(_translate("MainWindow", "阻断格数 z"))
        self.label_8.setText(_translate("MainWindow", "连接代价 g"))
        self.label_9.setText(_translate("MainWindow", "启发函数 h"))
        self.label_10.setText(_translate("MainWindow", "评价函数 f=g+h"))
        self.label_11.setText(_translate("MainWindow", "求解步骤"))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv) 
    w = QtWidgets.QMainWindow()
    ui = Ui_MainWindow() #类的实例
    ui.setupUi(w)
    w.show()
    sys.exit(app.exec_())