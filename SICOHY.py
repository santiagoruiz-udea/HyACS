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
import math
from xlsxwriter import Workbook                                     # Module for creating Excel XLSX files
from PyQt5 import QtCore, QtGui, uic, QtWidgets                     # Additional elements of PyQt5 module 
from PyQt5.QtWidgets import QMessageBox                             # Module to asking the user a question and receiving an answer
from PyQt5.QtGui import QCursor
import pandas as pd
from time import sleep
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from QtImageViewer import QtImageViewer
import matplotlib.pyplot as plt

""" ----------------------------------------------------------------------------------------------------------------------------------------------
    ----------------------------------- 2. Implementation of the class and its methods. ----------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------------- """
qtCreatorFile = "Interfaz.ui"                                       # Name of the GUI created using the Qt designer
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)          # The .ui file is imported to generate the graphical interface

qtCreatorFile = "results.ui"                                       # Name of the GUI created using the Qt designer
Ui_results, QtBaseClass = uic.loadUiType(qtCreatorFile)          # The .ui file is imported to generate the graphical interface

qtCreatorFile = "individuals.ui"                                  # Name of the GUI created using the Qt designer
Ui_individuals, QtBaseClass = uic.loadUiType(qtCreatorFile)           # The .ui file is imported to generate the graphical interface


class Results(QtWidgets.QMainWindow, Ui_results):
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

class Individuals(QtWidgets.QMainWindow, Ui_individuals):
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
    
    def __init__(self, dish, progressBar, val_start=0, val_max=15, delay=1.1, parent=None):
        super().__init__(parent)
        self.progressBar = progressBar
        self.value = val_start
        self.val_max = val_max
        self.delay = delay
        self.finish_flag = 0
        self.dish=dish
        self.scale_factor = 0.0

    def run(self):
        while self.finish_flag == 0:
            if self.value != self.val_max:
                self.value += 1
            
            sleep(self.delay)
            print("Progress {}%".format(self.value))
            self.progressBar.setValue(self.value)
                
        print('Progress bar completed!')
        self.finished.emit()
        
    def dish_edge(self):       
        gray = cv2.cvtColor(self.dish, cv2.COLOR_BGR2GRAY) 
        gray_blurred = cv2.blur(gray, (5, 5)) 
        
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 115, param2 = 115)#, minRadius = 1, maxRadius = 40) 
        detected_circles = np.uint16(np.around(detected_circles)) 
      
        pt = detected_circles[0,0]
        a, b, r = pt[0], pt[1], pt[2] 
     
        cv2.circle(self.dish, (a, b), r, (0, 255, 0), 10) 
        self.scale_factor = 100/(2*r)
                    
        print('Dish detected!')
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
        self.df = 0
        
        
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
            self.progressBar.hide()
        except:
            pass

    """-------------------------------------- c. Predictions --------------------------------------------------- """
    def run_barProgressUpdate(self, val_start, val_max, delay):
        
        "------------------------------------------------------------------------"
        self.thread_PB = QThread()
        self.thread_PB.daemon = False
        self.worker_PB = Worker(self.image, self.progressBar, val_start, val_max, delay)
        self.worker_PB.moveToThread(self.thread_PB)
        self.thread_PB.started.connect(self.worker_PB.run)
        self.worker_PB.finished.connect(self.thread_PB.quit)
        self.worker_PB.finished.connect(self.thread_PB.wait)
        self.worker_PB.finished.connect(self.worker_PB.deleteLater)
        self.thread_PB.finished.connect(self.thread_PB.deleteLater)
        self.thread_PB.start() 
        
    """-------------------------------------- c. Predictions --------------------------------------------------- """
    def run_detectDishEdge(self):
        
        "------------------------------------------------------------------------"
        self.thread_DE = QThread()
        self.thread_DE.daemon = False
        self.worker_DE = Worker(self.image, self.progressBar)
        self.worker_DE.moveToThread(self.thread_DE)
        self.thread_DE.started.connect(self.worker_DE.dish_edge)
        self.worker_DE.finished.connect(self.thread_DE.quit)
        self.worker_DE.finished.connect(self.thread_DE.wait)
        self.worker_DE.finished.connect(self.worker_DE.deleteLater)
        self.thread_DE.finished.connect(self.thread_DE.deleteLater)
        self.thread_DE.start() 

    """-------------------------------------- c. Predictions --------------------------------------------------- """
    def Start_execution(self):      
        self.progressBar.show() 
        self.run_barProgressUpdate(0, 85, 1.1)
        self.run_detectDishEdge()

        rows = []
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
        
        self.worker_PB.value = 85
        self.worker_PB.val_max = 99
        self.worker_PB.delay = 1
        
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
        cont = 0
        
        for i in range(len(boxes)):
            if i in self.indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0,255,0),3)
                cv2.putText(self.image, label, (x, y - 8), font, 0.4, (0,255,0), 1,cv2.LINE_AA)
                cv2.putText(self.image, str(round(confidences[i],2)), (x+50, y - 8), font, 0.4, (0,255,0), 1,cv2.LINE_AA)

                self.dic_hyallela[self.cont_hyallela] = [x, y, w, h]                                  # Add boxes in diccionary
                self.cont_hyallela += 1

                individual = self.img_copy[y-10:y+h+10,x-10:x+w+10,:].copy()
                
                # Binarizacion imagenes camara
                # hyallela_gray = cv2.cvtColor(individual,cv2.COLOR_BGR2GRAY)
                # ret, thresh = cv2.threshold(hyallela_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                
                # Binarizacion imagenes escaner
                hyallela_HSV= cv2.cvtColor(individual,cv2.COLOR_BGR2HSV)
                H,S,V = cv2.split(hyallela_HSV)
                hyallela_gray = cv2.cvtColor(individual,cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(H,30,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                
                thresh_copy = thresh.copy()
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))        #np.ones((3,3),np.uint8)                                    # Definition of the structural element for applying morphology
                morph = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        
                contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                contours = sorted(contours, key=cv2.contourArea,reverse=True)                                           
                
                try:
                    contorno_hyallela = contours[0]     
                except:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))        #np.ones((3,3),np.uint8)                                    # Definition of the structural element for applying morphology
                    morph = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
                    
                    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                    contours = sorted(contours, key=cv2.contourArea,reverse=True)  
                    contorno_hyallela = contours[0]     
                                                                    
                thresh[...] = 0                                                                             
                cv2.drawContours(thresh, [contorno_hyallela], 0, 255, cv2.FILLED)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                
                # cv2.imshow("thresh", thresh_copy)
                # cv2.imshow("thresh2", thresh2)
                # cv2.imshow("morph", morph)
                # cv2.imshow('individual', individual)
                # cv2.imshow('thresh after morph', thresh)
                # cv2.imshow('gray', hyallela_gray)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                
                #feature extraction
                area = cv2.contourArea(contours[0] )                                          # Area
                perimeter = round(cv2.arcLength(contours[0] ,True),2)                                  # Perimeter
                (x,y),(ma,MA),angle = cv2.fitEllipse(contours[0] )                            # Ellipse
                eccentricity = round(np.sqrt(1 - (pow(ma,2)/pow(MA,2))),2)           # Eccentricity
                
                # arc_length
                # Deleted noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
                
                # Find the area of the background
                sure_bg = cv2.dilate(opening,kernel,iterations=3)
                
                # Find the area of ​​the first
                dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

                # Find the unknown region (edges)
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg,sure_fg)
                img_no_int_part = unknown/255
                img_no_int_part = np.array(unknown, dtype=np.uint8)
                img_no_int_part = img_no_int_part * hyallela_gray
                
                alto, ancho = img_no_int_part.shape
                if alto*ancho<=4000:
                    Rmin=5
                elif alto*ancho<=9000:
                    Rmin=20
                elif alto*ancho<=12000:
                    Rmin=54
                elif alto*ancho<=20000:
                    Rmin = 59
                elif alto*ancho<=30000:
                    Rmin = 66
                else:
                    Rmin = 70

                detected_circles = cv2.HoughCircles(img_no_int_part, cv2.HOUGH_GRADIENT, 1.5, 20, param1 = 50, param2 = 30, minRadius = Rmin, maxRadius = 0)
                imagen_draw = thresh_copy*0
                
                if detected_circles is not None:
                    # Convert the circle parameters a, b and r to integers. 
                    detected_circles = np.uint16(np.around(detected_circles)) 
                    arc_length = []
                    for pt in detected_circles[0, :]: 
                        a, b, r = pt[0], pt[1], pt[2] 
                        imagen_draw = thresh_copy*0
                        # Draw the circumference of the circle. 
                        cv2.circle(imagen_draw, (a, b), r, 255, 1) 
                        and_img = cv2.bitwise_and(sure_bg,imagen_draw)
                        
                        #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                        #contours = sorted(contours, key=cv2.contourArea,reverse=True)                                           
                        #contorno_hyallela = contours[0]
                
                        arc_length.append(np.count_nonzero(and_img))

                    imagen_draw = thresh_copy*0
                    index_max_arc_length = np.argmax(arc_length)
                    cord = detected_circles[0][index_max_arc_length]
                    cv2.circle(imagen_draw, (cord[0], cord[1]), cord[2], 255, 1)
                    and_img = cv2.bitwise_and(sure_bg,imagen_draw)
                    
                    mask_rgb = np.zeros_like(individual)
                    mask_rgb[:,:,0] = and_img*0
                    mask_rgb[:,:,1] = and_img
                    mask_rgb[:,:,2] = and_img*0
                                
                    roi = individual.copy()
                    mask = cv2.bitwise_and(roi,roi,mask = 255-and_img)
                    
                    # cv2.imshow('mask', cv2.add(mask,mask_rgb))
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                        
                    arc_length = arc_length[index_max_arc_length]
                else:
                    cont +=1
                    (x,y),(a,b),angle = cv2.fitEllipse(contours[0])   
                    arc_length = (2/3)*math.pi*( 3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))/2
                
                # ---------------- Show result --------------------------
                x,y,w,h = cv2.boundingRect(contours[0])
                length = max(w,h)
                width = min(w,h)

                area = round(area*(self.worker_DE.scale_factor**2),2)
                perimeter = round(perimeter*self.worker_DE.scale_factor,2)
                arc_length = round(arc_length*self.worker_DE.scale_factor,2)
                length = round(length*self.worker_DE.scale_factor, 2)
                width = round(width*self.worker_DE.scale_factor,2)
                                
                row = [area, perimeter, eccentricity, arc_length, length, width]
                rows.append(row)

        print(cont)
        self.worker_PB.value = 100
        sleep(1.1)
        
        self.worker_PB.finish_flag = 1
        (myDirectory,nameFile) = os.path.split(self.image_path)
        self.df = pd.DataFrame(rows, columns=['Area (mm^2)','Perimeter (mm)','Eccentricity','Arc length (mm)', 'Length (mm)', 'Width (mm)'])

        area = self.df['Area (mm^2)'].values
        arc_length = self.df['Arc length (mm)'].values
        length = self.df['Length (mm)'].values
        width = self.df['Width (mm)'].values
        perimeter = self.df['Perimeter (mm)'].values
        eccentricity = self.df['Eccentricity'].values
        
        row = [[len(self.indexes), sum(area), round(sum(area)/len(area),2), round(sum(arc_length)/len(arc_length),2), round(sum(length)/len(length),2), round(sum(width)/len(width),2), round(sum(perimeter)/len(perimeter),2), round(sum(eccentricity)/len(eccentricity),2)]]
        dframe = pd.DataFrame(row, columns=['Individuals detected','Total area (mm^2)', 'Average area (mm^2)','Average Arc length (mm)', 'Average length (mm)','Average width (mm)','Average perimeter (mm)', 'Average eccentricity'])
 
        if (os.path.isdir(myDirectory + '/Result_' + nameFile[:-4]) == False):
            os.mkdir(myDirectory + '/Result_' + nameFile[:-4])
            
        if (os.path.isfile('Extracted_features.xlsx') == False):
            writer = pd.ExcelWriter(myDirectory + '/Result_' + nameFile[:-4] + '/Extracted_features.xlsx')
            self.df.to_excel(writer, sheet_name='Individuals',index=False) 
            dframe.to_excel(writer, sheet_name='Averages values',index=False) 
            writer.save()
            writer.close()
                   
        self.image_interface = cv2.resize(self.image, (720,720))                     # A video object is created according to the path selected
        cv2.imwrite(myDirectory + '/Result_' + nameFile[:-4] + '/labels.tif', self.image)
        
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
        self.dialog = Results(self)
        total_area = self.df['Area (mm^2)'].values
        length = self.df['Length (mm)'].values
        width = self.df['Width (mm)'].values            
        self.dialog.label_result.setText(str(len(self.indexes)))
        self.dialog.label_length.setText(str(round(sum(length)/len(length),2)) +' mm')
        self.dialog.label_width.setText(str(round(sum(width)/len(width),2)) +' mm')
        self.dialog.label_area.setText(str(round(sum(total_area),2)) +' mm^2')
        self.dialog.show()
        
    def Show_results_individual(self):
        self.dialog_individual = Individuals(self)
        self.dialog_individual.set_tamaño_items(self.cont_hyallela-1)
        self.dialog_individual.next.clicked.connect(self.show_hyallela)
        self.dialog_individual.previous.clicked.connect(self.show_hyallela)
        self.dialog_individual.show() 
        self.show_hyallela()
    
    def show_hyallela(self):
        x, y, w, h = self.dic_hyallela[self.dialog_individual.get_pos()] 
        individual = self.img_copy[y-10:y+h+10,x-10:x+w+10,:].copy()

        # Binarizacion imagenes camara
        # hyallela_gray = cv2.cvtColor(individual,cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(hyallela_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Binarizacion imagenes escaner
        hyallela_HSV= cv2.cvtColor(individual,cv2.COLOR_BGR2HSV)
        H,S,V = cv2.split(hyallela_HSV)
        hyallela_gray = cv2.cvtColor(individual,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(H,40,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        thresh_copy = thresh.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))        #np.ones((3,3),np.uint8)                                    # Definition of the structural element for applying morphology
        morph = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)                                           

        try:
            contorno_hyallela = contours[0]     
        except:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))        #np.ones((3,3),np.uint8)                                    # Definition of the structural element for applying morphology
            morph = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
            
            contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=cv2.contourArea,reverse=True)  
            contorno_hyallela = contours[0]   
        
        thresh[...] = 0                                                                             
        cv2.drawContours(thresh, [contorno_hyallela], 0, 255, cv2.FILLED)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
        cv2.drawContours(individual, contours, 0, (0,0,255), 1)

        # cv2.imshow("thresh", thresh_copy)
        # cv2.imshow("morph", morph)
        # cv2.imshow('individual', individual)
        # cv2.imshow('thresh after morph', thresh)
        # cv2.imshow('gray', hyallela_gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # arc_length
        # Deleted noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        
        # Find the area of the background
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        
        # Find the area of ​​the first
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

        # Find the unknown region (edges)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        img_no_int_part = unknown/255
        img_no_int_part = np.array(unknown, dtype=np.uint8)
        img_no_int_part = img_no_int_part * hyallela_gray
        
        alto, ancho = img_no_int_part.shape
        if alto*ancho<=4000:
            Rmin=5
        elif alto*ancho<=9000:
            Rmin=20
        elif alto*ancho<=12000:
            Rmin=54
        elif alto*ancho<=20000:
            Rmin = 59
        elif alto*ancho<=30000:
            Rmin = 66
        else:
            Rmin = 70

        detected_circles = cv2.HoughCircles(img_no_int_part, cv2.HOUGH_GRADIENT, 1.5, 20, param1 = 50, param2 = 30, minRadius = Rmin, maxRadius = 0)
        imagen_draw = thresh_copy*0
        
        if detected_circles is not None:
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles)) 
            arc_length = []
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2] 
                imagen_draw = thresh_copy*0
                # Draw the circumference of the circle. 
                cv2.circle(imagen_draw, (a, b), r, 255, 1) 
                and_img = cv2.bitwise_and(sure_bg,imagen_draw)        
                arc_length.append(np.count_nonzero(and_img))

            imagen_draw = thresh_copy*0
            index_max_arc_length = np.argmax(arc_length)
            cord = detected_circles[0][index_max_arc_length]
            cv2.circle(imagen_draw, (cord[0], cord[1]), cord[2], 255, 1)
            and_img = cv2.bitwise_and(sure_bg,imagen_draw)
            
            mask_rgb = np.zeros_like(individual)
            mask_rgb[:,:,0] = and_img*0
            mask_rgb[:,:,1] = and_img
            mask_rgb[:,:,2] = and_img*0
                        
            roi = individual.copy()
            mask = cv2.bitwise_and(roi,roi,mask = 255-and_img)
            individual = cv2.add(mask,mask_rgb)
            
            # cv2.imshow('mask', cv2.add(mask,mask_rgb))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
                
            arc_length = arc_length[index_max_arc_length]
        else:
            (x,y),(a,b),angle = cv2.fitEllipse(contours[0])   
            cv2.ellipse(individual,((x,y),(a,b),angle),(0,255,0),1)            
            arc_length = (2/3)*math.pi*( 3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))/2
            
        # ---------------- Show result --------------------------
        self.dialog_individual.label_area.setText(str(self.df.iloc[self.dialog_individual.get_pos()]['Area (mm^2)']) + ' mm^2')
        self.dialog_individual.label_perimeter.setText(str(self.df.iloc[self.dialog_individual.get_pos()]['Perimeter (mm)']) + ' mm')
        self.dialog_individual.label_eccentricity.setText(str(self.df.iloc[self.dialog_individual.get_pos()]['Eccentricity']))
        self.dialog_individual.label_cuvature.setText(str(self.df.iloc[self.dialog_individual.get_pos()]['Arc length (mm)']) + ' mm')
        self.dialog_individual.label_length.setText(str(self.df.iloc[self.dialog_individual.get_pos()]['Length (mm)']) + ' mm')
        self.dialog_individual.label_width.setText(str(self.df.iloc[self.dialog_individual.get_pos()]['Width (mm)']) + ' mm')

        self.image_interface = cv2.resize(individual, (461,271))                     # A video object is created according to the path selected
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
    

