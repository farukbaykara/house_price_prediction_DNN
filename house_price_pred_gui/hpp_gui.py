#!/usr/bin/env python3

from typing_extensions import Self
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn import model_selection

import hpp_predicter
import array as arr


class Ui_MainWindow():
    
    model_selection= 0
    MSSubClass = 0
    LotFrontage = 0
    LotArea = 0
    OverallQual = 0
    OverallCond = 0
    YearBuilt = 0
    YearRemodAdd = 0

    MassVnrArea = 0
    BsmtFinSF1 = 0
    BsmtFinSF2 = 0
    user_input = []*10


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(979, 793)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.predict_button = QtWidgets.QPushButton(self.centralwidget)
        self.predict_button.setGeometry(QtCore.QRect(570, 330, 181, 61))
        self.predict_button.setObjectName("predict_button")

        self.normal_model_choice = QtWidgets.QRadioButton(self.centralwidget)
        self.normal_model_choice.setGeometry(QtCore.QRect(570, 230, 112, 23))
        self.normal_model_choice.setObjectName("normal_model_choice")

        self.optimized_model_choice = QtWidgets.QRadioButton(self.centralwidget)
        self.optimized_model_choice.setGeometry(QtCore.QRect(570, 260, 271, 31))
        self.optimized_model_choice.setObjectName("optimized_model_choice")
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(570, 130, 321, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(570, 180, 401, 41))
        self.textBrowser.setObjectName("textBrowser")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 130, 111, 21))
        self.label_2.setObjectName("label_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 150, 371, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 190, 321, 21))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(40, 210, 271, 21))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(20, 250, 161, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(20, 270, 131, 21))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(20, 370, 161, 21))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(20, 390, 151, 16))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(20, 460, 161, 21))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(20, 440, 191, 16))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(20, 500, 171, 21))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(20, 520, 161, 21))
        self.label_15.setObjectName("label_15")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(20, 560, 171, 31))
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(20, 590, 67, 17))
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(20, 640, 67, 17))
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setGeometry(QtCore.QRect(20, 620, 301, 21))
        self.label_21.setObjectName("label_21")


        self.set_params_button = QtWidgets.QPushButton(self.centralwidget)
        self.set_params_button.setGeometry(QtCore.QRect(360, 730, 171, 41))
        self.set_params_button.setObjectName("set_params_button")


        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(20, 330, 151, 16))
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(20, 310, 161, 21))
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setGeometry(QtCore.QRect(20, 690, 67, 17))
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(self.centralwidget)
        self.label_25.setGeometry(QtCore.QRect(20, 670, 301, 21))
        self.label_25.setObjectName("label_25")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(280, 10, 411, 51))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(190, 60, 581, 61))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.predicted_price_outputLabel = QtWidgets.QLabel(self.centralwidget)
        self.predicted_price_outputLabel.setGeometry(QtCore.QRect(570, 410, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(17)


        self.predicted_price_outputLabel.setFont(font)
        self.predicted_price_outputLabel.setObjectName("predicted_price_outputLabel")


        self.MSSubClass_input_0 = QtWidgets.QLineEdit(self.centralwidget)
        self.MSSubClass_input_0.setGeometry(QtCore.QRect(390, 140, 113, 25))
        self.MSSubClass_input_0.setObjectName("MSSubClass_input_0")

        self.LotFrontage_input_1 = QtWidgets.QLineEdit(self.centralwidget)
        self.LotFrontage_input_1.setGeometry(QtCore.QRect(390, 200, 113, 25))
        self.LotFrontage_input_1.setObjectName("LotFrontage_input_1")


        self.LotArea_input_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.LotArea_input_2.setGeometry(QtCore.QRect(390, 260, 113, 25))
        self.LotArea_input_2.setObjectName("LotArea_input_2")


        self.OverallQual_input_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.OverallQual_input_3.setGeometry(QtCore.QRect(390, 320, 113, 25))
        self.OverallQual_input_3.setObjectName("OverallQual_input_3")


        self.OverallCond_input_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.OverallCond_input_4.setGeometry(QtCore.QRect(390, 380, 113, 25))
        self.OverallCond_input_4.setObjectName("OverallCond_input_4")


        self.YearBuilt_input_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.YearBuilt_input_5.setGeometry(QtCore.QRect(390, 450, 113, 25))
        self.YearBuilt_input_5.setObjectName("YearBuilt_input_5")


        self.YearRemodAdd_input_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.YearRemodAdd_input_6.setGeometry(QtCore.QRect(390, 510, 113, 25))
        self.YearRemodAdd_input_6.setObjectName("YearRemodAdd_input_6")


        self.MassVnrArea_input_7 = QtWidgets.QLineEdit(self.centralwidget)
        self.MassVnrArea_input_7.setGeometry(QtCore.QRect(390, 570, 113, 25))
        self.MassVnrArea_input_7.setObjectName("MassVnrArea_input_7")


        self.BsmtFinSF1_input_8 = QtWidgets.QLineEdit(self.centralwidget)
        self.BsmtFinSF1_input_8.setGeometry(QtCore.QRect(390, 630, 113, 25))
        self.BsmtFinSF1_input_8.setObjectName("BsmtFinSF1_input_8")


        self.BsmtFinSF2_input_9 = QtWidgets.QLineEdit(self.centralwidget)
        self.BsmtFinSF2_input_9.setGeometry(QtCore.QRect(390, 680, 113, 25))
        self.BsmtFinSF2_input_9.setObjectName("BsmtFinSF2_input_9")


        self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_3.setGeometry(QtCore.QRect(10, 10, 171, 31))
        self.textBrowser_3.setObjectName("textBrowser_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
#Prediction and user code ----------------------------------------------------------------
        

        self.set_params_button.clicked.connect(self.update_params)
        
        
        Ui_MainWindow.user_input.append(int(Ui_MainWindow.MSSubClass))
        Ui_MainWindow.user_input.append(int(Ui_MainWindow.LotFrontage))
        Ui_MainWindow.user_input.append(int(Ui_MainWindow.LotArea))
        Ui_MainWindow.user_input.append(int(Ui_MainWindow.OverallQual))
        Ui_MainWindow.user_input.append(int(Ui_MainWindow.OverallCond))
        Ui_MainWindow.user_input.append(int(Ui_MainWindow.YearBuilt))
        Ui_MainWindow.user_input.append(int(Ui_MainWindow.YearRemodAdd))
        Ui_MainWindow.user_input.append(int(Ui_MainWindow.MassVnrArea))
        Ui_MainWindow.user_input.append(int(Ui_MainWindow.BsmtFinSF1))
        Ui_MainWindow.user_input.append(int(Ui_MainWindow.BsmtFinSF2))


        self.optimized_model_choice.clicked.connect(self.tuned_model_selection)
        self.normal_model_choice.clicked.connect(self.normal_model_selection)

        
        
        self.predict_button.clicked.connect(self.output_screen_prediction)


        
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        

    def output_screen_prediction(self):        
        #_translate = QtCore.QCoreApplication.translate
        MSSubClass = self.MSSubClass_input_0.text()
        LotFrontage = self. LotFrontage_input_1.text()
        LotArea = self.LotArea_input_2.text()
        OverallQual = self.OverallQual_input_3.text()
        OverallCond = self.OverallCond_input_4.text()
        YearBuilt = self.YearBuilt_input_5.text()
        YearRemodAdd = self.YearRemodAdd_input_6.text()
        MassVnrArea = self.MassVnrArea_input_7.text()
        BsmtFinSF1 = self.BsmtFinSF1_input_8.text()
        BsmtFinSF2 = self.BsmtFinSF2_input_9.text()
        user_input = []*10
        user_input.append(int(MSSubClass))
        user_input.append(int(LotFrontage))
        user_input.append(int(LotArea))
        user_input.append(int(OverallQual))
        user_input.append(int(OverallCond))
        user_input.append(int(YearBuilt))
        user_input.append(int(YearRemodAdd))
        user_input.append(int(MassVnrArea))
        user_input.append(int(BsmtFinSF1))
        user_input.append(int(BsmtFinSF2))


        predicted_price = hpp_predicter.predict_price(user_input,Ui_MainWindow.model_selection)
        output_str = "Predicted Price :" + str(predicted_price) + "$"
        self.predicted_price_outputLabel.setText(output_str)



    def tuned_model_selection(self):
        Ui_MainWindow.model_selection = 1
    
    def normal_model_selection(self):
        Ui_MainWindow.model_selection = 0

    def update_params(self):
        Ui_MainWindow.MSSubClass = self.MSSubClass_input_0.text()
        Ui_MainWindow.LotFrontage = self. LotFrontage_input_1.text()
        Ui_MainWindow.LotArea = self.LotArea_input_2.text()
        Ui_MainWindow.OverallQual = self.OverallQual_input_3.text()
        Ui_MainWindow.OverallCond = self.OverallCond_input_4.text()
        Ui_MainWindow.YearBuilt = self.YearBuilt_input_5.text()
        Ui_MainWindow.YearRemodAdd = self.YearRemodAdd_input_6.text()
        Ui_MainWindow.MassVnrArea = self.MassVnrArea_input_7.text()
        Ui_MainWindow.BsmtFinSF1 = self.BsmtFinSF1_input_8.text()
        Ui_MainWindow.BsmtFinSF2 = self.BsmtFinSF2_input_9.text()


#---------------------------------------------------------------------------------
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.predict_button.setText(_translate("MainWindow", "Predict"))
        self.normal_model_choice.setText(_translate("MainWindow", "Model"))
        self.optimized_model_choice.setText(_translate("MainWindow", "Optimized Model"))
        self.label.setText(_translate("MainWindow", "Model (Weight) Selection"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Select model type which is optimized or not optimized. Prediction is made according to this selection. </p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Building Class:"))
        self.label_5.setText(_translate("MainWindow", "[60-20-70-50-190-45-90-120-30-85-80-160-75-180-40]"))
        self.label_6.setText(_translate("MainWindow", "Linear Feet of Street Connected to Property:"))
        self.label_7.setText(_translate("MainWindow", "Square feet area between [0-500]"))
        self.label_8.setText(_translate("MainWindow", "Lot size in suare feet :"))
        self.label_9.setText(_translate("MainWindow", "[1300 -  400000 ]"))
        self.label_10.setText(_translate("MainWindow", "Overall condition rating:"))
        self.label_11.setText(_translate("MainWindow", "[1-10]"))
        self.label_12.setText(_translate("MainWindow", "[1800 -  2022   ]"))
        self.label_13.setText(_translate("MainWindow", "Original construction date:"))
        self.label_14.setText(_translate("MainWindow", "Remodel date:"))
        self.label_15.setText(_translate("MainWindow", "[1800 -  2022   ]"))
        self.label_18.setText(_translate("MainWindow", "Masonry veneer area in square feet:"))
        self.label_19.setText(_translate("MainWindow", "[0- 1600]"))
        self.label_20.setText(_translate("MainWindow", "[0-2200]"))
        self.label_21.setText(_translate("MainWindow", "Basement type 1 finished area square feet:"))
        self.set_params_button.setText(_translate("MainWindow", "Set Input Parameters"))
        self.label_22.setText(_translate("MainWindow", "[1-10]"))
        self.label_23.setText(_translate("MainWindow", "Overall quality rating:"))
        self.label_24.setText(_translate("MainWindow", "[0-2200]"))
        self.label_25.setText(_translate("MainWindow", "Basement type 2 finished area square feet:"))
        self.label_3.setText(_translate("MainWindow", "House Price Prediction App"))
        self.textBrowser_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Write properties of your house which is you want to learn the predicted price of.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Set your parameters in given range and press the &quot;set input parameters&quot; button.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Also select the desired model type. Optimized model gives more accurate results.</p></body></html>"))
        self.predicted_price_outputLabel.setText(_translate("MainWindow", "Predicted Price :"))
        self.textBrowser_3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt;\">Faruk Baykara</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())