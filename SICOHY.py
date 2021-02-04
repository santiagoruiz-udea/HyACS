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
import PyQt5                                                        # Module of the set of Python bindings for Qt v5 
import numpy as np                                                  # Mathematical functions module
from xlsxwriter import Workbook                                     # Module for creating Excel XLSX files
from PyQt5 import QtCore, QtGui, uic, QtWidgets                     # Additional elements of PyQt5 module 
from PyQt5.QtWidgets import QMessageBox                             # Module to asking the user a question and receiving an answer
from PyQt5.QtGui import QCursor
import pandas as pd
from time import sleep
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from QtImageViewer import QtImageViewer


""" ----------------------------------------------------------------------------------------------------------------------------------------------
    ----------------------------------- 2. Implementation of the class and its methods. ----------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------------- """
qtCreatorFile = "Interfaz.ui"                                       # Name of the GUI created using the Qt designer
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)          # The .ui file is imported to generate the graphical interface

qtCreatorFile = "results.ui"                                       # Name of the GUI created using the Qt designer
Ui_results, QtBaseClass = uic.loadUiType(qtCreatorFile)          # The .ui file is imported to generate the graphical interface

qtCreatorFile = "individuals.ui"                                  # Name of the GUI created using the Qt designer
Ui_individuals, QtBaseClass = uic.loadUiType(qtCreatorFile)           # The .ui file is imported to generate the graphical interface


class Second(QtWidgets.QMainWindow, Ui_results):
    def __init__(self,  *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self,*args,**kwargs)
        self.setupUi(self)
        
        self.lb_udea.setMargin(3)                                     # Logo UdeA
        self.lb_gepar.setMargin(3)                                    # Logo GEPAR
        self.lb_geolimna.setMargin(3)                                 # Logo Capiro
        self.lb_gaia.setMargin(3)              

        self.Exit.clicked.connect(self.Exit_execution)              # Connection of the button with the Exit_execution method call


    def Exit_execution(self):
        """ This method has no input parameters and is responsible for definitely stopping frame capture by stopping the timer that controls 
            this process and exiting the MainWindow. Besides, this method stores the final results obtained in an Excel file and release the
            video input."""

        self.close()              # The graphical interface is closed

class Third(QtWidgets.QMainWindow, Ui_individuals):
    current_index = 0
    def __init__(self,  *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self,*args,**kwargs)
        self.setupUi(self)
               
        self.lb_udea.setMargin(3)                                     # Logo UdeA
        self.lb_gepar.setMargin(3)                                    # Logo GEPAR
        self.lb_geolimna.setMargin(3)                                 # Logo Capiro
        self.lb_gaia.setMargin(3)                                     # Logo Capiro

        self.tamaño_item = 0

        self.Exit.clicked.connect(self.Exit_execution)              # Connection of the button with the Exit_execution method call
        self.next.clicked.connect(self.next_img)                    # Connection of the button with the next_img method call   
        self.previous.clicked.connect(self.previous_img)            # Connection of the button with the previous_img method call   

    def next_img(self):
        if self.current_index == self.tamaño_item:
            self.current_index = 0
        else:
            self.current_index += 1

    def previous_img(self):
        if self.current_index == 0:
            self.current_index = self.tamaño_item
        else:
            self.current_index -= 1

    def set_tamaño_items(self,n):
        self.tamaño_item = n

    def get_pos(self):
        return self.current_index

    def Exit_execution(self):
        """ This method has no input parameters and is responsible for definitely stopping frame capture by stopping the timer that controls 
            this process and exiting the MainWindow. Besides, this method stores the final results obtained in an Excel file and release the
            video input."""

        self.close()              # The graphical interface is closed


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    
    def __init__(self, progressBar, val_start, val_max, delay, parent=None):
        super().__init__(parent)
        self.progressBar = progressBar
        self.value = val_start
        self.val_max = val_max
        self.delay = delay
        self.finish_flag = 0

    def run(self):
        while self.finish_flag == 0:
            if self.value != self.val_max:
                self.value += 1
            
            sleep(self.delay)
            print("Progress {}%".format(self.value))
            self.progressBar.setValue(self.value)
                
        print('Worker stopped!')
        self.finished.emit()

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
        self.img_copy = 0
        self.image_interface = 0
        self.lb_udea.setPixmap(QtGui.QPixmap('LogoUdea.png'))         # Logo UdeA
        self.lb_gepar.setPixmap(QtGui.QPixmap('LogoGepar.png'))       # Logo GEPAR
        self.lb_geolimna.setPixmap(QtGui.QPixmap('LogoGeoLimna.png')) # Logo GeoLimna
        self.lb_gaia.setPixmap(QtGui.QPixmap('LogoGaia.jpg'))         # Logo Gaia
        
        self.lb_udea.setMargin(3)                                     # Logo UdeA
        self.lb_gepar.setMargin(3)                                    # Logo GEPAR
        self.lb_geolimna.setMargin(3)                                 # Logo Capiro
        self.lb_gaia.setMargin(3)                                     # Logo Capiro
        
        self.Importar.clicked.connect(self.Import_image)                  # Connection of the button with the Import_image method call
        self.Start.clicked.connect(self.Start_execution)                  # Connection of the button with the Start_execution method call
        self.Results.clicked.connect(self.Show_results)                   # Connection of the button with the Show_results method call
        self.Individuals.clicked.connect(self.Show_results_individual)    # Connection of the button with the Show_results_individual call
        self.Exit.clicked.connect(self.Exit_execution)                    # Connection of the button with the Exit_execution method call
        self.setWindowIcon(QtGui.QIcon('udea.ico'))                       # Logo assignment (University's logo)
        self.frame_interface = QtImageViewer(self)
        
        self.frame_interface.hide()
        self.progressBar.hide()
        self.completed.hide()
        
        #Variables para calcular resultados
        self.cont_hyallela = 0                      # Dictionary's Key
        self.dic_hyallela = {}                      # Dictionary that holds anchor boxes 
        self.hyallela_lenght = 0                    # Hyallela's lenght
        self.hyallela_width = 0                     # Hyallela's width
        self.area = 0                               # Hyallela's area
        self.indexes = 0
        
        
    """-------------------------------------- b. Choice of directory for reading the video --------------------------------------------------- """
    def Import_image(self):
        """  In this method, the file browser is enabled so that the user selects the video that will be analyzed. Some variables are 
    	     initialized and it is started the timer that will control the capture of frames in the video to draw the line (In the preview
             of the video selected). This function doesn't return any variables."""
        try:
            # variable initialization
            self.cont_hyallela = 0                                 
            self.dic_hyallela = {}                            
    
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
            self.completed.hide()
        except:
            pass

    """-------------------------------------- c. Predictions --------------------------------------------------- """
    def run_barProgressUpdate(self, val_start, val_max, delay):
        
        "------------------------------------------------------------------------"
        self.thread = QThread()
        self.thread.daemon = False
        self.worker = Worker(self.progressBar, val_start, val_max, delay)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.thread.wait)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start() 

    """-------------------------------------- c. Predictions --------------------------------------------------- """
    def Start_execution(self):      
        self.progressBar.show() 
        self.run_barProgressUpdate(0, 85, 1.1)

        net=  cv2.dnn.readNet('SICOHY.weights', 'SICOHY.cfg')        
        classes = []
        with open('SICOHY.names',"r") as f:
          classes = [line.strip() for line in f.readlines()]
                
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        #self.image = cv2.resize(self.image, None, fx=0.3, fy = 0.3)
        height,width,channels =self.image.shape
        self.img_copy = self.image.copy()
        
        # Detecting objects
        blob = cv2.dnn.blobFromImage(self.image, 0.00392,(32*64,32*64),(0,0,0), True, crop= False)
        net.setInput(blob)
        outs = net.forward(output_layers) #Caracteristica layers
        
        self.worker.value = 85
        self.worker.val_max = 99
        self.worker.delay = 0.4
        
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
        
        self.indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # Si se repite la identificacion es decir hay dos centro en un mismo objeto
        font = cv2.FONT_HERSHEY_DUPLEX
        
        for i in range(len(boxes)):
            if i in self.indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0,255,0),2)
                cv2.putText(self.image, label, (x, y - 8), font, 0.4, (0,255,0), 1,cv2.LINE_AA)
                cv2.putText(self.image, str(round(confidences[i],2)), (x+50, y - 8), font, 0.4, (0,255,0), 1,cv2.LINE_AA)
                self.dic_hyallela[self.cont_hyallela] = [x, y, w, h]                                  # Add boxes in diccionary
                self.cont_hyallela += 1
        
        self.worker.value = 100
        sleep(1.2)
        self.worker.finish_flag = 1
        
        #cv2.imwrite('/content/drive/Shared drives/8_SICOHY: Sistema de identificación, clasificación y conteo Hyalellas/2_Experimentos_Software/4_Base_de_datos/Test.tif', img)
        
        print(len(self.indexes))
                   
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

        self.Individuals.setStyleSheet("color: white; background:  rgb(39, 108, 222); border: 1px solid white; border-radius: 10px; font: 75 14pt 'Reboto Medium';")
        self.Individuals.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.completed.show()

        
    """------------------------------------------------ g.  Stopping the execution  --------------------------------------------------------"""       
    def Show_results(self):
        self.dialog = Second(self)
        self.dialog.label_result.setText(str(len(self.indexes)))
        #self.dialog.label_length.setText(str(round(sum(self.average_lenght)/len(self.average_lenght),2)) +' px')
        #self.dialog.label_width.setText(str(round(sum(self.average_width)/len(self.average_width),2)) +' px')
        #self.dialog.label_area.setText(str(self.area) +' px')
        self.dialog.show()
        
    def Show_results_individual(self):
        self.dialog_individual = Third(self)
        self.dialog_individual.set_tamaño_items(self.cont_hyallela-1)
        self.dialog_individual.next.clicked.connect(self.show_hyallela)
        self.dialog_individual.previous.clicked.connect(self.show_hyallela)
        self.dialog_individual.show() 
    
    def show_hyallela(self):
        x, y, w, h = self.dic_hyallela[self.dialog_individual.get_pos()] 
        hyallela_img = self.img_copy[y-10:y+h+10,x-10:x+w+10,:]
        hyallela_img_copy = hyallela_img.copy()
        hyallela_gray = cv2.cvtColor(hyallela_img,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(hyallela_gray,200,255,cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)                                           
        contorno_hyallela = contours[0]                                                                         
        thresh[...] = 0                                                                             
        cv2.drawContours(thresh, [contorno_hyallela], 0, 255, cv2.FILLED)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        thresh_copy = thresh.copy()
        #feature extraction
        cnt = contours[0]                                                    # Hyallela's Contours
        area = cv2.contourArea(cnt)                                          # Area
        perimeter = cv2.arcLength(cnt,True)                                  # Perimeter
        (x,y),(ma,MA),angle = cv2.fitEllipse(cnt)                            # Ellipse
        eccentricity = round(np.sqrt(1 - (pow(ma,2)/pow(MA,2))),2)           # Eccentricity

        # Curvature
        imagen_draw = hyallela_gray*0
        src = cv2.medianBlur(hyallela_img_copy, 5)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        try:
            circles = np.uint16(np.around(circles))
            cord = circles[0][0]
            cv2.circle(imagen_draw, (cord[0], cord[1]), cord[2], 255, 1)
            and_img = cv2.bitwise_and(thresh_copy,imagen_draw)
            curvature = np.count_nonzero(and_img)
        except:
            curvature = 'none'
        
        # ---------------- Show result --------------------------
        # Major axis like length and minor axis like width
        if (2*MA > 2*ma):
            self.dialog_individual.label_length.setText(str(int(2*MA)))
            self.dialog_individual.label_width.setText(str(int(2*ma)))
        # Major axis like width and minor axis like length
        else:
            self.dialog_individual.label_length.setText(str(int(2*ma)))
            self.dialog_individual.label_width.setText(str(int(2*MA)))

        self.dialog_individual.label_area.setText(str(area))
        self.dialog_individual.label_perimeter.setText(str(perimeter))
        self.dialog_individual.label_eccentricity.setText(str(eccentricity))
        self.dialog_individual.label_cuvature.setText(str(curvature))

        self.image_interface = cv2.resize(hyallela_img, (461,271))                     # A video object is created according to the path selected
        image_interface = QtGui.QImage(self.image_interface,self.image_interface.shape[1],self.image_interface.shape[0],self.image_interface.shape[1]*self.image_interface.shape[2],QtGui.QImage.Format_RGB888)
        frame_interface = QtGui.QPixmap()
        frame_interface.convertFromImage(image_interface.rgbSwapped())

        self.dialog_individual.Lb_hyallela.setPixmap(frame_interface)
        self.dialog_individual.Lb_hyallela.show()

         

    """------------------------------------------------ h. Exiting the execution  --------------------------------------------------------"""
    def Exit_execution(self):
        """ This method has no input parameters and is responsible for definitely stopping frame capture by stopping the timer that controls 
            this process and exiting the MainWindow. Besides, this method stores the final results obtained in an Excel file and release the
            video input."""

        window.close()              # The graphical interface is closed


# Main implementation      
if __name__ == "__main__":
    
    dirname = os.path.dirname(PyQt5.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    QtWidgets.QApplication.addLibraryPath(plugin_path)
    app =  QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    

