# -*- coding: utf-8 -*-
""" ----------------------------------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------ GEPAR, GEOLIMNA and GAIA research groups ----------------------------------------------------
    -------------------------------------------------------- University of Antioquia -------------------------------------------------------------
    ----------------------------------------------------------- Medellín, Colombia ---------------------------------------------------------------
    ------------------------------------------------------------- February, 2021 -----------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------------------------------------------------
    --------------------------------------------- Authors: * David Stephen Fernández Mc Cann -----------------------------------------------------
    ------------------------------------------------------ * Fabio de Jesús Vélez Macias ---------------------------------------------------------
    ------------------------------------------------------ * Nestor Jaime Aguirre Ramírez --------------------------------------------------------
    ------------------------------------------------------ * Julio Eduardo Cañón Barriga ---------------------------------------------------------
    ------------------------------------------------------ * Ludy Yanith Pineda Alarcón ----------------------------------------------------------
    ------------------------------------------------------ * Yarin Tatiana Puerta Quintana -------------------------------------------------------
    ------------------------------------------------------ * Maycol EstebanZuluaga Montoya -------------------------------------------------------
    ------------------------------------------------------ * Santiago Ruiz González --------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------------------------------------------------
    ------------------ Project Name: Identification, counting and metrics calculation of Hyalella individuals inside a Petri dish ----------------
    ----------------------------------------------------------------------------------------------------------------------------------------------
    --------Description: This project aims to implement an algorithm to detect Hyalella individuals using images of Petri dishes, in -------------
    -------------------- order to count the amount of individuals, and subsequently, calculate some metrics associated with each -----------------
    -------------------- organism such as arc length, area, etc. This was possible by applying artificial vision techniques, specifically --------
    -------------------- the CNN called YOLO (You Only Look Once), which was used to obtain a model that is imported here. In the interface ------
    -------------------- the user is able to import a Petri dish image and after processing it is given to they the ammount of individuals, ------
    -------------------- different metrics, a summary of these metrics, and a Excel file where this information is also available. ---------------
    ---------------------------------------------------------------------------------------------------------------------------------------------- """



""" ----------------------------------------------------------------------------------------------------------------------------------------------
    --------------------------------------------------- 1. Libraries needed ----------------------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------------- """
import os                                                           # Operating system dependent functionalities module
import cv2                                                          # Images processing module 
import sys                                                          # Module of variables and functions used by the interpreter
import PyQt5                                                        # Module of the set of Python bindings for Qt v5 
import numpy as np                                                  # Mathematical functions module
import pandas as pd                                                 # Data structures and data analysis tools

from time import sleep                                              # Suspends execution of the current thread for a given number of seconds.
from PyQt5.QtGui import QCursor                                     # Class to change the cursor style
from QtImageViewer import QtImageViewer                             # Module to use the QtImageViewer which includes zoom functionality
from PyQt5.QtWidgets import QMessageBox                             # Module to show pop-up messages
from PyQt5 import QtCore, QtGui, uic, QtWidgets                     # Additional elements of PyQt5 module 
from PyQt5.QtCore import QObject, QThread, pyqtSignal               # Thread using tools from QT



""" ----------------------------------------------------------------------------------------------------------------------------------------------
    ----------------------------------------- 2. Classes implementation and its methods. ---------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------------- """
qtCreatorFile = "Interfaz.ui"                                       # Name of the mainwindow GUI created using the Qt designer
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)          # The .ui file is imported to generate the graphical interface

qtCreatorFile = "results.ui"                                        # Name of the general results GUI created using the Qt designer
Ui_results, QtBaseClass = uic.loadUiType(qtCreatorFile)             

qtCreatorFile = "individuals.ui"                                    # Name of the individual results GUI created using the Qt designer
Ui_individuals, QtBaseClass = uic.loadUiType(qtCreatorFile)         


""" ----------------------------------------- 2.1. Implementation of the general results class ------------------------------------------ """
class Results(QtWidgets.QMainWindow, Ui_results):
    """--------------------- 2.1.a. Constructor ---------------------------------"""
    def __init__(self,  *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self,*args,**kwargs)
        self.setupUi(self)
        self.lb_udea.setMargin(3)                                     # Logo UdeA margin
        self.lb_gepar.setMargin(3)                                    # Logo GEPAR margin
        self.lb_geolimna.setMargin(3)                                 # Logo Geolimna margin
        self.lb_gaia.setMargin(3)                                     # Logo Gaia margin
        self.Exit.clicked.connect(self.Exit_execution)                # Connection of the button with the Exit_execution method call

    """---------------------------- 2.1.b. Exit ---------------------------------"""
    def Exit_execution(self):
        self.close()                                                  # The graphical interface is closed


""" --------------------------------------- 2.2. Implementation of the individual results class ----------------------------------------- """
class Individuals(QtWidgets.QMainWindow, Ui_individuals):
    """--------------------- 2.2.a. Constructor --------------------------------"""
    def __init__(self,  *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self,*args,**kwargs)
        self.setupUi(self)    
        self.amount_items = 0                                        # Size (h*w) of each individual to restrict the max radius in Hough transform
        self.current_index = 0                                       # Index to select which individual will be shown 
        self.lb_udea.setMargin(3)                                    # Logo UdeA
        self.lb_gepar.setMargin(3)                                   # Logo GEPAR
        self.lb_geolimna.setMargin(3)                                # Logo Capiro
        self.lb_gaia.setMargin(3)                                    # Logo Capiro
        self.Exit.clicked.connect(self.Exit_execution)               # Connection of the button with the Exit_execution method call
        self.next.clicked.connect(self.next_img)                     # Connection of the button with the next_img method call   
        self.previous.clicked.connect(self.previous_img)             # Connection of the button with the previous_img method call   

    """---------- 2.2.b. Method to go to the next image ------------------------"""
    def next_img(self):
        if self.current_index == self.amount_items:
            self.current_index = 0
        else:
            self.current_index += 1

    """------------- 2.2.c. Method to go to the previous image ------------------"""
    def previous_img(self):
        if self.current_index == 0:
            self.current_index = self.amount_items
        else:
            self.current_index -= 1

    """------- 2.2.d. Method to set the amount of individuals founded ------------"""
    def set_amount_items(self,amount):
        self.amount_items = amount

    """------- 2.2.e. Method to get the current index of individual shown ---------"""
    def get_pos(self):
        return self.current_index

    """------- 2.2.f. Method to exit the dialog -----------------------------------"""
    def Exit_execution(self):
        self.close()              



""" --------------------------------------- 2.3. Implementation of the class worker --------------------------------------- """
class Worker(QObject):
    finished = pyqtSignal()           # Signal to emit when worker is done
    
    """---------------------- 2.3.a. Constructor ---------------------------------"""
    def __init__(self, dish, progressBar, val_start=0, val_max=15, delay=1.1, parent=None):
        super().__init__(parent)
        self.progressBar = progressBar
        self.current_value = val_start
        self.val_max = val_max
        self.delay = delay
        self.finish_flag = 0
        self.dish=dish
        self.scale_factor = 0.0

    """---------------------- 2.3.b. ProgressBar update loop ----------------------"""
    def run(self):
        while self.finish_flag == 0:
            if self.current_value != self.val_max:              
                self.current_value += 1
            sleep(self.delay)                                       # Sleep during the time indicated
            self.progressBar.setValue(self.current_value)           # ProgressBar update

        self.finished.emit()                                        # Finish signal emition when the flag is 1
       
    """------------------- 2.3.c. Petri dish border detection ---------------------"""
    def dish_edge(self):
        try:       
            gray = cv2.cvtColor(self.dish, cv2.COLOR_BGR2GRAY)      # Image in grayscale
            gray_blurred = cv2.blur(gray, (5, 5))                   # Image blurring
            
            # Detect circles using the Hough transformation
            detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 115, param2 = 115)
            detected_circles = np.uint16(np.around(detected_circles)) 
            pt = detected_circles[0,0]
            a, b, r = pt[0], pt[1], pt[2]                           # Parameters of the circle founded
            cv2.circle(self.dish, (a, b), r, (0, 255, 0), 10)       # The circle is drawn 
            self.scale_factor = 100/(2*r)                           # Petri dish's diameter = 100 mm
            self.finished.emit()                                    # Finish signal emition
        except:
            pass


""" --------------------------------------- 2.4. Implementation of the class MainWindow --------------------------------------- """
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    """------------------------- 2.4.a. Constructor -------------------------------"""
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self,*args,**kwargs)
        self.setupUi(self)

        # Atributes
        self.image_path = 0
        self.image = 0
        self.img_copy = 0
        self.image_interface = 0
        self.lb_udea.setPixmap(QtGui.QPixmap('LogoUdea.png'))             # Logo UdeA
        self.lb_gepar.setPixmap(QtGui.QPixmap('LogoGepar.png'))           # Logo GEPAR
        self.lb_geolimna.setPixmap(QtGui.QPixmap('LogoGeoLimna.png'))     # Logo GeoLimna
        self.lb_gaia.setPixmap(QtGui.QPixmap('LogoGaia.jpg'))             # Logo Gaia
        self.lb_udea.setMargin(3)                                         # Logo UdeA margin
        self.lb_gepar.setMargin(3)                                        # Logo GEPAR margin
        self.lb_geolimna.setMargin(3)                                     # Logo Capiro margin
        self.lb_gaia.setMargin(3)                                         # Logo Capiro margin
        
        self.Importar.clicked.connect(self.Import_image)                  # Connection of the button with the Import_image method call
        self.Start.clicked.connect(self.Start_execution)                  # Connection of the button with the Start_execution method call
        self.Results.clicked.connect(self.Show_results)                   # Connection of the button with the Show_results method call
        self.Individuals.clicked.connect(self.Show_results_individual)    # Connection of the button with the Show_results_individual call
        self.Exit.clicked.connect(self.Exit_execution)                    # Connection of the button with the Exit_execution method call
        self.setWindowIcon(QtGui.QIcon('udea.ico'))                       # Logo assignment (University's logo)
        self.frame_interface = QtImageViewer(self) 
                
        self.cont_hyallela = 0                                            # Dictionary's keys
        self.dic_hyallela = {}                                            # Dictionary that holds bounding boxes of each Hyalella 
        self.hyallela_lenght = 0                                          # Hyallela's lenght
        self.hyallela_width = 0                                           # Hyallela's width
        self.area = 0                                                     # Hyallela's area
        self.indexes = 0
        self.individuals = 0
        
        self.frame_interface.hide()                                       # It is hidden the QtImageViewer where will be put the images
        self.progressBar.hide()                                           # It is hidden the Progress bar
        self.completed.hide()                                             # It is hidden the 'completed' mesagge
        
        
    """--------------- 2.4.b. Import the image thath will be processed ------------"""
    def Import_image(self):
        try:
            self.image_path, _ = QtWidgets.QFileDialog.getOpenFileName(self)                # File explorer is opened to select the image that will be used                   
            self.image = cv2.imread(self.image_path)                                        # The image is read 
            self.image_interface = cv2.resize(self.image, (720,720))                        # Resizing to show the image in the interface
    
            image_interface=QtGui.QImage(self.image_interface,self.image_interface.shape[1],self.image_interface.shape[0],self.image_interface.shape[1]*self.image_interface.shape[2],QtGui.QImage.Format_RGB888)
            frame_interface = QtGui.QPixmap()
            frame_interface.convertFromImage(image_interface.rgbSwapped())
    
            self.frame_interface.setImage(frame_interface)                                  # The image is set in the QtImageViewer
            self.frame_interface.setGeometry(30, 50, 635, 635)                              # The QtImageViewer is resized
            self.frame_interface.setStyleSheet("background:  rgb(39, 108, 222); border: 1px solid rgb(39, 108, 222)")
            self.frame_interface.show()
            self.completed.hide()                                                           # It is hidden the 'completed' message
            self.progressBar.hide()                                                         # It is hidden the Progress bar
        except:
            pass

    """--------- 2.4.c. New thread to run the progressBar update process ---------"""
    def run_barProgressUpdate(self, val_start, val_max, delay):
        
        self.thread_PB = QThread()                                                         # New QThread instance
        self.worker_PB = Worker(self.image, self.progressBar, val_start, val_max, delay)   # New worker instance
        self.worker_PB.moveToThread(self.thread_PB)                                        # The worker is added to the thread
        self.thread_PB.started.connect(self.worker_PB.run)                                 # Start method connection for the thread
        self.worker_PB.finished.connect(self.thread_PB.quit)                               # Finish methods connection for the worker
        self.worker_PB.finished.connect(self.thread_PB.wait)
        self.worker_PB.finished.connect(self.worker_PB.deleteLater)
        self.thread_PB.finished.connect(self.thread_PB.deleteLater)                        # Finish methods connection for the thread
        self.thread_PB.start()                                                             # The thread is started
        
    """---------- 2.4.d. New thread to run the Petri dish edge detection process ---------"""
    def run_detectDishEdge(self):
        
        self.thread_DE = QThread()                                                         # New QThread instance
        self.worker_DE = Worker(self.image, self.progressBar)                              # New worker instance
        self.worker_DE.moveToThread(self.thread_DE)                                        # The worker is added to the thread
        self.thread_DE.started.connect(self.worker_DE.dish_edge)                           # Start method connection for the thread
        self.worker_DE.finished.connect(self.thread_DE.quit)                               # Finish methods connection for the worker
        self.worker_DE.finished.connect(self.thread_DE.wait)
        self.worker_DE.finished.connect(self.worker_DE.deleteLater)
        self.thread_DE.finished.connect(self.thread_DE.deleteLater)                        # Finish methods connection for the thread
        self.thread_DE.start()                                                             # The thread is started


    """----- 2.4.e. Start the process to detect Hyalellas and calculate their metrics -----"""
    def Start_execution(self):
        try:      
            self.progressBar.show()                                                         # The progress bar appers in the interface
            self.run_barProgressUpdate(0, 85, 1.1)                                          # It is started the thread to update the progressBar till 85%
            self.run_detectDishEdge()                                                       # It is started the thread to detect the Petri dish edge and the scale factor
            self.img_copy = self.image.copy()
            classes = []                                                                    # Classes of the YOLO model
            boxes = []                                                                      # Boxes of the predictions with a confidence over 0.5
            class_ids = []                                                                  # Id of each prediction
            confidences = []                        
            excel_file_rows = []
    
            # --------------- YOLO model setting-----------------------
            net = cv2.dnn.readNet('HyCAS.weights', 'HyCAS.cfg')                             # The model is read (weights and configuration file)
            with open('HyCAS.names',"r") as f:                                              # Classes are read
                classes = [line.strip() for line in f.readlines()]                          
                    
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            height,width,channels =self.image.shape
            
            # -------------- Detecting objects -------------------------
            blob = cv2.dnn.blobFromImage(self.image, 0.00392,(32*64,32*64),(0,0,0), True, crop= False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            self.worker_PB.current_value = 85                                               # So far it's done about 85%
            self.worker_PB.val_max = 99                                                     # The max_value is set to 99
            self.worker_PB.delay = 1
                
            # ------ Loop to keep just the prediction with more than 50% confidence ------         
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:
                        # Object detected coordinates
                        center_x = int(detection[0]*width)                                  # Center's x coordinate
                        center_y = int(detection[1]*height)                                 # Center's y coordinate
                        w = int(detection[2]*width)                                         # Object width                                 
                        h = int(detection[3]*height)                                        # Object length
                        x = int(center_x - w /2)                                            # Upper left corner x coordinate
                        y = int(center_y - h /2)                                            # Upper left corner y coordinate
                        
                        boxes.append([x, y, w, h])                                          # Object coordinates added to the list of all bounding boxes
                        confidences.append(float(confidence))                               # Object confidnece added to the list of all confidences
                        class_ids.append(class_id)                                          # Object id added to the list of all ids
                                    
            self.indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)                   # To avoid 2 centers in the same object
            font = cv2.FONT_HERSHEY_DUPLEX  
    
            # ------- Loop to draw all of the objects detected in the original image ---------
            for i in range(len(boxes)):
                if i in self.indexes:
                    x, y, w, h = boxes[i]                                                                # Extract the coordinates of the bounding box
                    label = str(classes[class_ids[i]])                                                   # Name of the class detected
                    cv2.rectangle(self.image, (x, y), (x + w, y + h), (0,255,0),3)                       # It is drawn the bounding box
                    cv2.putText(self.image, label, (x, y - 8), font, 0.4, (0,255,0), 1,cv2.LINE_AA)      # Puts the text of the class label set
                    cv2.putText(self.image, str(round(confidences[i],2)), (x+50, y - 8), font, 0.4, (0,255,0), 1,cv2.LINE_AA)
                    self.dic_hyallela[self.cont_hyallela] = [x, y, w, h]                                 # Add boxes in diccionary
                    self.cont_hyallela += 1                                                              # Increase the hyalellas counter
    
                    # ---- Binarization ------
                    individual = self.img_copy[y-10:y+h+10,x-10:x+w+10,:].copy()                                # Copy of the region where is the hyalella
                    hyallela_HSV= cv2.cvtColor(individual,cv2.COLOR_BGR2HSV)                                    # Color space HSV
                    H,S,V = cv2.split(hyallela_HSV)                                                             # Splits the 3 channels
                    hyallela_gray = cv2.cvtColor(individual,cv2.COLOR_BGR2GRAY)                                 # Image of the individual in gray scale
                    ret, thresh = cv2.threshold(H,30,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)                 # Binarization
                    
                    # ------ Morphology to smooth the contour ------
                    thresh_copy = thresh.copy()
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))                                 # Structural element for applying morphology
                    morph = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)                       # Open morphology operation
                    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)       # Contours are founded
                    contours = sorted(contours, key=cv2.contourArea,reverse=True)                                # Contours are sorted by area
                    
                    try:
                        contorno_hyallela = contours[0]     
                    except:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))                             # Definition of the structural element for applying morphology
                        morph = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)                   # Open morphology operation
                        contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)   # Contours are founded
                        contours = sorted(contours, key=cv2.contourArea,reverse=True)                            # Contours are sorted by area
                        contorno_hyallela = contours[0]                                                          # The contour with the bigger area is selected
                                                                        
                    thresh[...] = 0                                                                             
                    cv2.drawContours(thresh, [contorno_hyallela], 0, 255, cv2.FILLED)                            # The contour is drawn
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)      # It is founded the contour again
                                        
                    # ---- Feature extraction ----
                    area = cv2.contourArea(contours[0] )                                          # Area
                    perimeter = round(cv2.arcLength(contours[0] ,True),2)                         # Perimeter
                    (x,y),(ma,MA),angle = cv2.fitEllipse(contours[0] )                            # Ellipse
                    eccentricity = round(np.sqrt(1 - (pow(ma,2)/pow(MA,2))),2)                    # Eccentricity
                    
                    # ---------- Arc length calculation -----------
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))                  # Structural element
                    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)      # Open morphology operation
                    sure_bg = cv2.dilate(opening,kernel,iterations=3)                             # Find the area of the background
                    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)                 # Image with the distance from white pixels to the nearest black pixel
                    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)   # Binarization with the internal line
    
                    # ------ Find the unknown region (edges) ------
                    sure_fg = np.uint8(sure_fg)
                    unknown = cv2.subtract(sure_bg,sure_fg)
                    img_no_int_part = unknown/255
                    img_no_int_part = np.array(unknown, dtype=np.uint8)
                    img_no_int_part = img_no_int_part * hyallela_gray
                    
                    # ------ Conditional to set the minimum radio of the circles detected ------
                    height, width = img_no_int_part.shape
                    if height*width<=4000:
                        Rmin=5
                    elif height*width<=9000:
                        Rmin=20
                    elif height*width<=12000:
                        Rmin=54
                    elif height*width<=20000:
                        Rmin = 59
                    elif height*width<=30000:
                        Rmin = 66
                    else:
                        Rmin = 70
    
                    # ---- Hough transformation to detect the circles in the contours
                    detected_circles = cv2.HoughCircles(img_no_int_part, cv2.HOUGH_GRADIENT, 1.5, 20, param1 = 50, param2 = 30, minRadius = Rmin, maxRadius = 0)
                    imagen_draw = thresh_copy*0
                    
                    if detected_circles is not None:
                        detected_circles = np.uint16(np.around(detected_circles))              # Convert the circle parameters a, b and r to integers. 
                        arc_length = []                                                        # Vector to store the arc lengths  
                        
                        # Loop to calculate the arc length usign each circle detected
                        for pt in detected_circles[0, :]: 
                            a, b, r = pt[0], pt[1], pt[2]                                      # Center and radius
                            imagen_draw = thresh_copy*0                                        # Image to draw the circle
                            cv2.circle(imagen_draw, (a, b), r, 255, 1)                         # Draw the circumference of the circle.       
                            and_img = cv2.bitwise_and(sure_bg,imagen_draw)                     # And bewteen the contour and the circle
                            arc_length.append(np.count_nonzero(and_img))                       # The arc length is the sum of the non-zero pixels
                            
                        imagen_draw = thresh_copy*0
                        index_max_arc_length = np.argmax(arc_length)                           # It is selected the index of the bigger arc length founded
                        cord = detected_circles[0][index_max_arc_length]                       # Coordinates of the circle thath provides the bigger arc length
                        cv2.circle(imagen_draw, (cord[0], cord[1]), cord[2], 255, 1)           # It is drawn the circle
                        and_img = cv2.bitwise_and(sure_bg,imagen_draw)                         # And to get the intersection between the contour and the circle
                        arc_length = arc_length[index_max_arc_length]                          # Arc length in pixels
                        arc_length = round(arc_length*self.worker_DE.scale_factor,2)           # Arc length in milimeters    
                    else:
                        arc_length = 'Not founded'
                                        
                    # ---------------- Remaining metrics -----------------------
                    x,y,w,h = cv2.boundingRect(contours[0])
                    length = max(w,h)                                                          # Length in pixels
                    length = round(length*self.worker_DE.scale_factor, 2)                      # Length in milimeters
                    width = min(w,h)                                                           # Width in pixels
                    width = round(width*self.worker_DE.scale_factor,2)                         # Width in milimeters
                    area = round(area*(self.worker_DE.scale_factor**2),2)                      # Area
                    perimeter = round(perimeter*self.worker_DE.scale_factor,2)                 # Perimeter
                                    
                    row = [area, perimeter, eccentricity, arc_length, length, width]           # Row to add to the excel file
                    excel_file_rows.append(row)                                                # Row added to the excel file
                     
            self.worker_PB.current_value = 100                                                 # Process done, progress bar is set in 100%
            sleep(1.1)                                                                         # Sleep to update the interface
            self.worker_PB.finish_flag = 1                                                     # Flag to stop the thread
            (myDirectory,nameFile) = os.path.split(self.image_path)
            self.individuals = pd.DataFrame(excel_file_rows, columns=['Area (mm^2)','Perimeter (mm)','Eccentricity','Arc length (mm)', 'Length (mm)', 'Width (mm)'])
    
            # ----- Information extracted from the dataframe -------
            area = self.individuals['Area (mm^2)'].values                           
            arc_length = self.individuals['Arc length (mm)'].values
            arc_length = np.where(arc_length=='Not founded', np.nan,arc_length)
            length = self.individuals['Length (mm)'].values
            width = self.individuals['Width (mm)'].values
            perimeter = self.individuals['Perimeter (mm)'].values
            eccentricity = self.individuals['Eccentricity'].values
            
            # ---- Row to store the average values ------
            row = [[len(self.indexes), sum(area), round(area.mean(),2), round(np.nanmean(arc_length),2), round(length.mean(),2), round(width.mean(),2), round(perimeter.mean(),2), round(eccentricity.mean(),2)]]
            averages = pd.DataFrame(row, columns=['Individuals detected','Total area (mm^2)', 'Average area (mm^2)','Average Arc length (mm)', 'Average length (mm)','Average width (mm)','Average perimeter (mm)', 'Average eccentricity'])
    
            if (os.path.isdir(myDirectory + '/Result_' + nameFile[:-4]) == False):                              # It is created the folder to store the results
                os.mkdir(myDirectory + '/Result_' + nameFile[:-4])      
                
            cv2.imwrite(myDirectory + '/Result_' + nameFile[:-4] + '/labels.tif', self.image)                   # Writes the image with the bounding boxes drawn
            
            if (os.path.isfile('Extracted_features.xlsx') == False):                                            # It is created the excel file to store the results
                writer = pd.ExcelWriter(myDirectory + '/Result_' + nameFile[:-4] + '/Extracted_features.xlsx')
                self.individuals.to_excel(writer, sheet_name='Individuals',index=False)                         # Sheet of the individuals values
                averages.to_excel(writer, sheet_name='Average values',index=False)                              # Sheet of the average values
                writer.save()
                writer.close()
               
            # ------ The image with the results is placed in the interface ---------
            self.image_interface = cv2.resize(self.image, (720,720))                                            
            image_interface=QtGui.QImage(self.image_interface,self.image_interface.shape[1],self.image_interface.shape[0],self.image_interface.shape[1]*self.image_interface.shape[2],QtGui.QImage.Format_RGB888)
            frame_interface = QtGui.QPixmap()
            frame_interface.convertFromImage(image_interface.rgbSwapped())
            self.frame_interface.setImage(frame_interface)
            self.frame_interface.setGeometry(30, 50, 635, 635)
            self.frame_interface.setStyleSheet("background:  rgb(39, 108, 222); border: 1px solid rgb(39, 108, 222)")
            self.frame_interface.show()
            
            # ------ The button to visualize the average results is enabled ----------
            self.Results.setStyleSheet("color: white; background:  rgb(39, 108, 222); border: 1px solid white; border-radius: 10px; font: 75 14pt 'Reboto Medium';")
            self.Results.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
    
            # ----- The button to visualize the individual results is enabled --------
            self.Individuals.setStyleSheet("color: white; background:  rgb(39, 108, 222); border: 1px solid white; border-radius: 10px; font: 75 14pt 'Reboto Medium';")
            self.Individuals.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
            self.completed.show()
        except:
            self.worker_PB.finish_flag = 1
            self.progressBar.close()
            
            if self.image_path == 0 :
                # It indicates that a image has not been imported yet 
                message = 'The image has not been imported, please do so to be able to extract information.'
                QMessageBox.setStyleSheet(self,"QMessageBox\n{\n	background-color: rgb(255, 255, 255);\n}\n")
                QMessageBox.critical(self, 'Start error', message)
            else:
                # It indicates that the imported image is invalid
                message = 'The file selected is invalid, please import a valid file to extract information'
                QMessageBox.setStyleSheet(self,"QMessageBox\n{\n	background-color: rgb(255, 255, 255);\n}\n")
                QMessageBox.critical(self, 'Import error', message)
                self.image_path = 0

        
    """------------------- 2.4.f. Average results visualization -------------------"""
    def Show_results(self):
        # ----- Information extracted from the dataframe -------
        area = self.individuals['Area (mm^2)'].values
        length = self.individuals['Length (mm)'].values
        width = self.individuals['Width (mm)'].values   
        arc_length = self.individuals['Arc length (mm)'].values
        arc_length = np.where(arc_length=='Not founded', np.nan,arc_length)
        perimeter = self.individuals['Perimeter (mm)'].values
        eccentricity = self.individuals['Eccentricity'].values
        
        # ----- Set text to each label -------
        self.dialog = Results(self)        
        self.dialog.label_individuals.setText(str(len(self.indexes)))
        self.dialog.label_length.setText(str(round(length.mean(),2)) +' mm')
        self.dialog.label_width.setText(str(round(width.mean(),2)) +' mm')
        self.dialog.label_arc_length.setText(str(round(np.nanmean(arc_length),2)) + ' mm')
        self.dialog.label_perimeter.setText(str(round(perimeter.mean(),2)) + ' mm')
        self.dialog.label_eccentricity.setText(str(round(eccentricity.mean(),2)))
        self.dialog.label_aver_area.setText(str(round(area.mean(),2)) + ' mm^2')
        self.dialog.label_area.setText(str(round(sum(area),2)) +' mm^2')
        self.dialog.show()
        
    """------------------- 2.4.g. Individual results visualization -------------------"""
    def Show_results_individual(self):
        self.dialog_individual = Individuals(self)                              # Instance of the class
        self.dialog_individual.set_amount_items(self.cont_hyallela-1)           # Set the amount of Hyalellas founded
        self.dialog_individual.next.clicked.connect(self.show_hyallela)         # Connect the next click with function 'show_hyallela'
        self.dialog_individual.previous.clicked.connect(self.show_hyallela)     # Connect the previous click with function 'show_hyallela'
        self.dialog_individual.show()                                           # Shows the dialogs
        self.show_hyallela()                                    
    
    """------------------- 2.4.g. Individual Hyalella visualization -------------------"""
    def show_hyallela(self):
        x, y, w, h = self.dic_hyallela[self.dialog_individual.get_pos()]                         # Bounding box values of the current hyalella 
        individual = self.img_copy[y-10:y+h+10,x-10:x+w+10,:].copy()                             # Mask where it is located the individual

        # ------ Binarization ------
        hyallela_HSV= cv2.cvtColor(individual,cv2.COLOR_BGR2HSV)                                 # Color space HSV
        H,S,V = cv2.split(hyallela_HSV)                                                          # Splits the 3 channels
        hyallela_gray = cv2.cvtColor(individual,cv2.COLOR_BGR2GRAY)                              # Image of the individual in gray scale
        ret, thresh = cv2.threshold(H,40,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)              # Binarization
        
        # ------ Morphology to smooth the contour ------
        thresh_copy = thresh.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))                             # Structural element for applying morphology
        morph = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)                   # Open morphology operation
        contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)   # Contours are founded
        contours = sorted(contours, key=cv2.contourArea,reverse=True)                            # Contours are sorted by area                      

        try:
            contorno_hyallela = contours[0]     
        except:      
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))                             # Definition of the structural element for applying morphology
            morph = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)                   # Open morphology operation
            contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)   # Contours are founded
            contours = sorted(contours, key=cv2.contourArea,reverse=True)                            # Contours are sorted by area
            contorno_hyallela = contours[0]                                                          # The contour with the bigger area is selected
                          
        thresh[...] = 0                                                                             
        cv2.drawContours(thresh, [contorno_hyallela], 0, 255, cv2.FILLED)                            # The contour is drawn
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)      # It is founded the contour again
        cv2.drawContours(individual, contours, 0, (0,0,255), 1)                                      # The contour (perimeter) is drawn 
        
        # ---------- Arc length calculation -----------
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))                  # Structural element
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)      # Open morphology operation
        sure_bg = cv2.dilate(opening,kernel,iterations=3)                             # Find the area of the background
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)                 # Image with the distance from white pixels to the nearest black pixel
        ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)   # Binarization with the internal line

        # ------ Find the unknown region (edges) ------
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        img_no_int_part = unknown/255
        img_no_int_part = np.array(unknown, dtype=np.uint8)
        img_no_int_part = img_no_int_part * hyallela_gray
        
        # ---- Conditional to set the minimum radio of the circles detected -----
        height, width = img_no_int_part.shape
        if height*width<=4000:
            Rmin=5
        elif height*width<=9000:
            Rmin=20
        elif height*width<=12000:
            Rmin=54
        elif height*width<=20000:
            Rmin = 59
        elif height*width<=30000:
            Rmin = 66
        else:
            Rmin = 70
            
        # ---- Hough transformation to detect the circles in the contours
        detected_circles = cv2.HoughCircles(img_no_int_part, cv2.HOUGH_GRADIENT, 1.5, 20, param1 = 50, param2 = 30, minRadius = Rmin, maxRadius = 0)
        imagen_draw = thresh_copy*0
        
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))              # Convert the circle parameters a, b and r to integers. 
            arc_length = []                                                        # Vector to store the arc lengths 
            
            # Loop to calculate the arc length usign each circle detected
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2]                                      # Center and radius
                imagen_draw = thresh_copy*0                                        # Image to draw the circle
                cv2.circle(imagen_draw, (a, b), r, 255, 1)                         # Draw the circumference of the circle.       
                and_img = cv2.bitwise_and(sure_bg,imagen_draw)                     # And bewteen the contour and the circle
                arc_length.append(np.count_nonzero(and_img))                       # The arc length is the sum of the non-zero pixels
           
            # --- Detection of the line intersection
            imagen_draw = thresh_copy*0
            index_max_arc_length = np.argmax(arc_length)                           # It is selected the index of the bigger arc length founded
            cord = detected_circles[0][index_max_arc_length]                       # Coordinates of the circle thath provides the bigger arc length
            cv2.circle(imagen_draw, (cord[0], cord[1]), cord[2], 255, 1)           # It is drawn the circle
            and_img = cv2.bitwise_and(sure_bg,imagen_draw)                         # And to get the intersection between the contour and the circle

            # --- Mask with the line intersection drawn in green
            mask_rgb = np.zeros_like(individual)
            mask_rgb[:,:,0] = and_img*0
            mask_rgb[:,:,1] = and_img
            mask_rgb[:,:,2] = and_img*0
                        
            roi = individual.copy()                                                 # Takes the mask of the individual location
            mask = cv2.bitwise_and(roi,roi,mask = 255-and_img)                      # Individual location with pixels zero pixels in the intersection line
            individual = cv2.add(mask,mask_rgb)                                     # The red line is added to full the zero pixels 
            arc_length = arc_length[index_max_arc_length]                           
        else:           
            arc_length = 'Not founded'
            
        # ----- Set text to each label -------
        self.dialog_individual.label_area.setText(str(self.individuals.iloc[self.dialog_individual.get_pos()]['Area (mm^2)']) + ' mm^2')
        self.dialog_individual.label_perimeter.setText(str(self.individuals.iloc[self.dialog_individual.get_pos()]['Perimeter (mm)']) + ' mm')
        self.dialog_individual.label_eccentricity.setText(str(self.individuals.iloc[self.dialog_individual.get_pos()]['Eccentricity']))
        self.dialog_individual.label_length.setText(str(self.individuals.iloc[self.dialog_individual.get_pos()]['Length (mm)']) + ' mm')
        self.dialog_individual.label_width.setText(str(self.individuals.iloc[self.dialog_individual.get_pos()]['Width (mm)']) + ' mm')
        if self.individuals.iloc[self.dialog_individual.get_pos()]['Arc length (mm)'] == 'Not founded':
            self.dialog_individual.label_cuvature.setText(str(self.individuals.iloc[self.dialog_individual.get_pos()]['Arc length (mm)']))
        else:
            self.dialog_individual.label_cuvature.setText(str(self.individuals.iloc[self.dialog_individual.get_pos()]['Arc length (mm)']) + ' mm')

        # ------ The individual image with the arc length and perimeter is placed in the interface ---------
        self.image_interface = cv2.resize(individual, (461,271))                     
        image_interface = QtGui.QImage(self.image_interface,self.image_interface.shape[1],self.image_interface.shape[0],self.image_interface.shape[1]*self.image_interface.shape[2],QtGui.QImage.Format_RGB888)
        frame_interface = QtGui.QPixmap()
        frame_interface.convertFromImage(image_interface.rgbSwapped())
        self.dialog_individual.Lb_hyallela.setPixmap(frame_interface)
        self.dialog_individual.Lb_hyallela.show()


    """------------------- 2.4.h. Method to exit the dialog -----------------------------------"""
    def Exit_execution(self):
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