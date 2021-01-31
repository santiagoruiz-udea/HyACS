# -*- coding: utf-8 -*-
""" ----------------------------------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------ GEPAR and GeoLimna research groups ----------------------------------------------------------
    ----------------------------------------------------- University of Antioquia ----------------------------------------------------------------
    ------------------------------------------------------- Medellín, Colombia -------------------------------------------------------------------
    -------------------------------------------------------- February, 2021 ---------------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------------------------------------------------
    --------------------------------------------- Authors: * David Stephen Fernández Mc Cann -----------------------------------------------------
    ------------------------------------------------------ * Fabio de Jesús Vélez Macias ---------------------------------------------------------
    ------------------------------------------------------ * Nestor Jaime Aguirre Ramírez --------------------------------------------------------
    ------------------------------------------------------ * Santiago Ruiz González --------------------------------------------------------------
    ------------------------------------------------------ * Maycol Zuluaga Montoya --------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------------------------------------------------
    ------------ Project Name:  ------------------------
    ----------------------------------------------------------------------------------------------------------------------------------------------
    ----------Description:  -------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------------- """

""" ----------------------------------------------------------------------------------------------------------------------------------------------
    --------------------------------------------- 1. Import of the necessary libraries -----------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------------- """
import os                                                           # Operating system dependent functionalities module
import cv2                                                          # Images processing module 
import sys                                                          # Module of variables and functions used by the interpreter
import math                                                         # C standard mathematical functions module
import PyQt5                                                        # Module of the set of Python bindings for Qt v5 
import matplotlib                                                   # Plotting module 
import numpy as np                                                  # Mathematical functions module
import matplotlib.pyplot as plt                                     # Module that provides a MATLAB-like plotting framework 
from xlsxwriter import Workbook                                     # Module for creating Excel XLSX files
from skimage.measure import label, regionprops                      # Additional images processing module
from PyQt5 import QtCore, QtGui, uic, QtWidgets                     # Additional elements of PyQt5 module 
from PyQt5.QtWidgets import QMessageBox                             # Module to asking the user a question and receiving an answer
from PyQt5.QtGui import QCursor
import pandas as pd
import joblib
from skimage.measure import label, regionprops
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
from PyQt5.QtWidgets import QPushButton,QVBoxLayout


import sys
try:
    from PyQt5.QtCore import Qt, QT_VERSION_STR
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QApplication, QFileDialog
except ImportError:
    try:
        from PyQt4.QtCore import Qt, QT_VERSION_STR
        from PyQt4.QtGui import QImage, QApplication, QFileDialog
    except ImportError:
        raise ImportError("Requires PyQt5 or PyQt4.")
from QtImageViewer import QtImageViewer


""" ----------------------------------------------------------------------------------------------------------------------------------------------
    ----------------------------------- 2. Implementation of the class and its methods. ----------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------------- """
qtCreatorFile = "Interfaz.ui"                                       # Name of the GUI created using the Qt designer
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)          # The .ui file is imported to generate the graphical interface

qtCreatorFile = "results.ui"                                       # Name of the GUI created using the Qt designer
Ui_results, QtBaseClass = uic.loadUiType(qtCreatorFile)          # The .ui file is imported to generate the graphical interface

class Second(QtWidgets.QMainWindow, Ui_results):
    def __init__(self,  *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self,*args,**kwargs)
        self.setupUi(self)
        
        self.lb_udea.setMargin(3)                                   # Logo UdeA
        self.lb_gepar.setMargin(3)                                  # Logo GEPAR
        self.lb_capiro.setMargin(3)                                 # Logo Capiro
        
# Implementation of the class MainWindow
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    """---------------------------------------------- a. Constructor ---------------------------------------------------"""
    def __init__(self, *args, **kwargs):
        """ This method is the class constructor, where the necessary variables for the correct functioning of the software
    	    are defined as attributes. Also here, the icon is configured and the images of the logos are assigned to the
	        Labels established for it. Finally, the interface buttons are connected with their associated functions. """

        QtWidgets.QMainWindow.__init__(self,*args,**kwargs)
        self.setupUi(self)

        # Atributes
        self.image_path = 0
        self.image = 0
        self.image_interface = 0
        self.lb_udea.setPixmap(QtGui.QPixmap('LogoUdea.png'))         # Logo UdeA
        self.lb_gepar.setPixmap(QtGui.QPixmap('LogoGepar.png'))       # Logo GEPAR
        self.lb_geolimna.setPixmap(QtGui.QPixmap('LogoGeoLimna.png')) # Logo GeoLimna
        self.lb_gaia.setPixmap(QtGui.QPixmap('LogoGaia.jpg'))         # Logo Gaia

        
        self.lb_udea.setMargin(3)                                     # Logo UdeA
        self.lb_gepar.setMargin(3)                                    # Logo GEPAR
        self.lb_geolimna.setMargin(3)                                 # Logo Capiro
        self.lb_gaia.setMargin(3)                                 # Logo Capiro
        
        self.Importar.clicked.connect(self.Import_image)            # Connection of the button with the Import_image method call
        self.Start.clicked.connect(self.Start_execution)            # Connection of the button with the Start_execution method call
        self.Results.clicked.connect(self.Show_results)          # Connection of the button with the Show_results method call
        self.Exit.clicked.connect(self.Exit_execution)              # Connection of the button with the Exit_execution method call
        self.setWindowIcon(QtGui.QIcon('udea.ico'))                 # Logo assignment (University's logo)
        self.frame_interface = QtImageViewer(self)
        
        self.frame_interface.hide()
        #Variables para calcular resultados
        self.cont_incestos = 0
        self.average_lenght = []
        self.average_width = []
        self.area = 0
        
        
    """-------------------------------------- b. Choice of directory for reading the video --------------------------------------------------- """
    def Import_image(self):
        """  In this method, the file browser is enabled so that the user selects the video that will be analyzed. Some variables are 
    	     initialized and it is started the timer that will control the capture of frames in the video to draw the line (In the preview
             of the video selected). This function doesn't return any variables."""
            
        self.image_path, _ = QtWidgets.QFileDialog.getOpenFileName(self)   # File explorer is opened to select the video that will be used                     
        self.image = cv2.imread(self.image_path)                     # A video object is created according to the path selected
        self.image_interface = cv2.resize(self.image, (720,720))                     # A video object is created according to the path selected

        image_interface=QtGui.QImage(self.image_interface,self.image_interface.shape[1],self.image_interface.shape[0],self.image_interface.shape[1]*self.image_interface.shape[2],QtGui.QImage.Format_RGB888)
        frame_interface = QtGui.QPixmap()
        frame_interface.convertFromImage(image_interface.rgbSwapped())

        self.frame_interface.setImage(frame_interface)
        self.frame_interface.setGeometry(30, 50, 635, 635)
        self.frame_interface.setStyleSheet("background:  rgb(39, 108, 222); border: 1px solid rgb(39, 108, 222)")
        self.frame_interface.show()

    """-------------------------------------- c. Predictions --------------------------------------------------- """
    def Start_execution(self):      
        print('Processing...')
        net=  cv2.dnn.readNet('SICOHY.weights', 'SICOHY.cfg')
        
        classes = []
        with open('SICOHY.names',"r") as f:
          classes = [line.strip() for line in f.readlines()]
                
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        #self.image = cv2.resize(self.image, None, fx=0.3, fy = 0.3)
        height,width,channels =self.image.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(self.image, 0.00392,(32*64,32*64),(0,0,0), True, crop= False)
        net.setInput(blob)
        outs = net.forward(output_layers) #Caracteristica layers
        
        # Showing infromations on the screen
        confidences = []
        boxes = []
        class_ids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    #object detected
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w /2)
                    y = int(center_y - h /2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
                   # cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0),2)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # Si se repite la identificacion es decir hay dos centro en un mismo objeto
        font = cv2.FONT_HERSHEY_DUPLEX
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0,255,0),2)
                cv2.putText(self.image, label, (x, y - 8), font, 0.4, (0,255,0), 1,cv2.LINE_AA)
                cv2.putText(self.image, str(round(confidences[i],2)), (x+50, y - 8), font, 0.4, (0,255,0), 1,cv2.LINE_AA)
        
        #cv2.imwrite('/content/drive/Shared drives/8_SICOHY: Sistema de identificación, clasificación y conteo Hyalellas/2_Experimentos_Software/4_Base_de_datos/Test.tif', img)
        
        print(len(indexes))
                   
        #cv2.imwrite(self.image_path[:-4]+'_labeled.JPG', img)
        self.image_interface = cv2.resize(self.image, (720,720))                     # A video object is created according to the path selected

        image_interface=QtGui.QImage(self.image_interface,self.image_interface.shape[1],self.image_interface.shape[0],self.image_interface.shape[1]*self.image_interface.shape[2],QtGui.QImage.Format_RGB888)
        frame_interface = QtGui.QPixmap()
        frame_interface.convertFromImage(image_interface.rgbSwapped())

        self.frame_interface.setImage(frame_interface)
        self.frame_interface.setGeometry(30, 50, 635, 635)
        self.frame_interface.setStyleSheet("background:  rgb(39, 108, 222); border: 1px solid rgb(39, 108, 222)")
        self.frame_interface.show()
        
        self.Results.setStyleSheet("color: white; background:  rgb(39, 108, 222); border: 1px solid white; border-radius: 10px; font: 75 14pt 'Reboto Medium';")
        self.Results.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        
    """------------------------------------------------ g.  Stopping the execution  --------------------------------------------------------"""       
    def Show_results(self):
        self.dialog = Second(self)
        self.dialog.label_result.setText(str(self.cont_incestos))
        self.dialog.label_length.setText(str(round(sum(self.average_lenght)/len(self.average_lenght),2)) +' px')
        self.dialog.label_width.setText(str(round(sum(self.average_width)/len(self.average_width),2)) +' px')
        self.dialog.label_area.setText(str(self.area) +' px')
        self.dialog.show()
        self.dialog.Exit.clicked.connect(self.Exit_dialog)
        
    
         

    """------------------------------------------------ h. Exiting the execution  --------------------------------------------------------"""
    def Exit_execution(self):
        """ This method has no input parameters and is responsible for definitely stopping frame capture by stopping the timer that controls 
            this process and exiting the MainWindow. Besides, this method stores the final results obtained in an Excel file and release the
            video input."""

        window.close()              # The graphical interface is closed

    """------------------------------------------------ i. Exiting the Dialog    --------------------------------------------------------"""
    def Exit_dialog(self):
        """ This method has no input parameters and is responsible for definitely stopping frame capture by stopping the timer that controls 
            this process and exiting the Dialog. Besides, this method stores the final results obtained in an Excel file and release the
            video input."""
        self.dialog.close()

# Main implementation      
if __name__ == "__main__":
    
    dirname = os.path.dirname(PyQt5.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    QtWidgets.QApplication.addLibraryPath(plugin_path)
    app =  QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    

