from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5 import Qt
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget,QTableWidget,QTableWidgetItem,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
from Design import Ui_MainWindow

import os
import io as _io
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from xlrd import open_workbook
from openpyxl.reader.excel import load_workbook

from skimage import data, img_as_float,io
from skimage.measure import compare_ssim as SSIM2

from sklearn.metrics import mean_squared_error as MSE
from PIL import Image
import scipy
import pandas as pd
from scipy import ndimage
from sklearn import decomposition
from skimage import data
from skimage import color
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.measure import regionprops
import random
import matplotlib.image as IMG
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from PIL.ImageQt import ImageQt
from skimage.feature import daisy
import pickle
from sklearn.metrics import jaccard_similarity_score
from skimage.feature import daisy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class override_graphicsScene (Qt.QGraphicsScene):
    def __init__(self,parent = None):
        super(override_graphicsScene,self).__init__(parent)

    def mousePressEvent(self, event):
        super(override_graphicsScene, self).mousePressEvent(event)
        print(event.pos())

class MainWindow(QWidget,Ui_MainWindow):
    
    deger_path = "image.png"
    deger_path_2 = "image_2.png"
    deger_path_3 = "image_3.png"
    
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)


        

        
        self.btn_keypoint_referance_points.clicked.connect(self.button_keypoint_referance_points)
        self.btn_keypoint_apply.clicked.connect(self.SURF_UYGULA)
        self.btn_keypoint_apply_2.clicked.connect(self.siftuygula)
        self.btn_keypoint_apply_3.clicked.connect(self.orbuygula)

        

    def ALGORITHM_RANDOM_FOREST(self,_x_train,_x_test,_y_train,_y_test):
        model = RandomForestClassifier()
        model.fit(_x_train,_y_train)
        y_pred = model.predict(_x_test)
        accurity = accuracy_score(_y_test,y_pred)
        return str(round(accurity*100,2))

    def build_filters(self):
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
        return filters
         
    def gabor_features(self,img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
        return accum

    #KEY POİNT VERİ YÜKLEME
    dosyaismi = ""
    dosyadataset = []
    def button_keypoint_referance_points(self):
        file,_ = QFileDialog.getOpenFileName(self, 'Open file', './',"Pkl Files (*.pkl *.)")
        self.dosyaismi = file
        self.dosyadataset=[]
        
        with open(self.dosyaismi, 'rb') as f:
            data = pickle.load(f)
        
        self.table_keypoint_referans.setColumnCount(len(data["image001.png"])+1)
        self.table_keypoint_referans.setRowCount(len(data))
        for i,row in enumerate(data):
            referance_temp=[]
            referance_temp.append(row)
            self.table_keypoint_referans.setItem(i,0, QTableWidgetItem(str(row)))
            for j, value in enumerate(data[row]):
                self.table_keypoint_referans.setItem(i,j+1, QTableWidgetItem(str(value)))
                referance_temp.append(value)
            
            self.dosyadataset.append(referance_temp)
        
        self.table_keypoint_referans.horizontalHeader().setStretchLastSection(True)
        self.table_keypoint_referans.resizeColumnsToContents()
        self.table_keypoint_referans.setHorizontalHeaderLabels(self.label_header_keypoint)

        
        #print(self.dosyadataset)
        
    
    directory_retina_source = "./objects/retina/sources/"
    directory_retina_result = "./objects/retina/results/"
    directory_retina_threshold = "./objects/retina/threshold/"
    directory_retina_clahe = "./objects/retina/clahe/"
    directory_retina_sobel = "./objects/retina/sobel/"

    label_header_keypoint = ['Image','y','x','size','Best Score','Jaccard']
    
    
    def orbuygula(self):        
        self.table_keypoint_result_3.clear()        
        results2=[]
        count_point2=0        
        dec2 = cv2.ORB_create(nfeatures=1500)
        for i,value in enumerate(self.dosyadataset):
            deger_img_source = cv2.imread(self.directory_retina_source+value[0])
            img_referance = cv2.imread(self.directory_retina_source+value[0],cv2.COLOR_BGR2GRAY)
            img_referance = self.resimkes(img_referance,value[2],value[1],value[3])
            img_source = cv2.imread(self.directory_retina_source+value[0],cv2.COLOR_BGR2GRAY)
            img_referance = self.step_I(img_referance,None)
            img_source = self.step_I(img_source,self.directory_retina_clahe+value[0])
            keypoints, descriptors = dec2.detectAndCompute(img_source, None)
            best_score = 0
            jaccard_score = 0
            gecicix,geciciy,gecicir = self.dosyadataset[i][2],self.dosyadataset[i][1],self.dosyadataset[i][3]
            deger_name,deger_x,deger_y,deger_r='',0,0,0
            deger_img = 0            
            for index in range(len(keypoints)):
                count_point2 += 1
                x = int(keypoints[index].pt[0])
                y = int(keypoints[index].pt[1])
                r = value[3]                            
                img_source_point = self.resimkes(img_source,x,y,r)             
                if(img_referance.shape == img_source_point.shape):
                    score = self.ssim2(img_referance,img_source_point)#jaccard_similarity_score
                    if(score>best_score):
                        best_score=score
                        jaccard_score= self.jaccard(img_referance,img_source_point)
                        deger_img = img_source#img_source_point
                        deger_name = value[0]
                        deger_x,deger_y,deger_r=x,y,r
            
            results2.append([resimad,ynoktası,xnoktası,genişlik,sonucdegeri,jaccarddegeri])
            deger_img=cv2.rectangle(deger_img_source,(gecicix-gecicir,geciciy-gecicir),(gecicix+gecicir,geciciy+gecicir),(255,0,0),3)            
            deger_img=cv2.rectangle(deger_img,(deger_x-deger_r,deger_y-deger_r),(deger_x+deger_r,deger_y+deger_r),(255,0,0),3)
            cv2.imwrite(self.directory_retina_result+deger_name,deger_img)

        self.table_keypoint_result_3.setColumnCount(len(results2[0]))
        self.table_keypoint_result_3.setRowCount(len(results2))
        for i,row in enumerate(results2):
            for j, value in enumerate(row):
                self.table_keypoint_result_3.setItem(i,j, QTableWidgetItem(str(value)))
        
        self.table_keypoint_result_3.horizontalHeader().setStretchLastSection(True)
        self.table_keypoint_result_3.resizeColumnsToContents()
        self.table_keypoint_result_3.setHorizontalHeaderLabels(self.label_header_keypoint)
           
        
        #1. resim
        image_name = results2[0][0]
        scene = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img1.size())
        self.img1.setScene(scene)        
        jaccard_score =str(results2[0][4])
        self.jacdeger1.setText(str(jaccard_score))
         #2. resim
        image_name = results2[1][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img2.size())
        self.img2.setScene(scene2)        
        jaccard_score =str(results2[1][4])
        self.jacdeger2.setText(str(jaccard_score))           
         #3. resim
        image_name = results2[2][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img3.size())
        self.img3.setScene(scene2)        
        jaccard_score =str(results2[2][4])
        self.jacdeger3.setText(str(jaccard_score)) 
         #4. resim
        image_name = results2[3][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img4.size())
        self.img4.setScene(scene2)        
        jaccard_score =str(results2[3][4])
        self.jacdeger4.setText(str(jaccard_score))         
         #5. resim
        image_name = results2[4][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img5.size())
        self.img5.setScene(scene2)        
        jaccard_score =str(results2[4][4])
        self.jacdeger5.setText(str(jaccard_score)) 
         #6. resim
        image_name = results2[5][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img6.size())
        self.img6.setScene(scene2)        
        jaccard_score =str(results2[5][4])
        self.jacdeger6.setText(str(jaccard_score)) 
         #7. resim
        image_name = results2[6][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img7.size())
        self.img7.setScene(scene2)        
        jaccard_score =str(results2[6][4])
        self.jacdeger7.setText(str(jaccard_score)) 
         #8. resim
        image_name = results2[7][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img8.size())
        self.img8.setScene(scene2)        
        jaccard_score =str(results2[7][4])
        self.jacdeger8.setText(str(jaccard_score)) 
         #9. resim
        image_name = results2[9][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img9.size())
        self.img9.setScene(scene2)        
        jaccard_score =str(results2[9][4])
        self.jacdeger9.setText(str(jaccard_score)) 
         #10. resim
        image_name = results2[9][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img10.size())
        self.img10.setScene(scene2)        
        jaccard_score =str(results2[9][4])
        self.jacdeger10.setText(str(jaccard_score)) 
        
         #11. resim
        image_name = results2[10][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img11.size())
        self.img11.setScene(scene2)        
        jaccard_score =str(results2[10][4])
        self.jacdeger11.setText(str(jaccard_score)) 
         #12. resim
        image_name = results2[11][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img12.size())
        self.img12.setScene(scene2)        
        jaccard_score =str(results2[11][4])
        self.jacdeger12.setText(str(jaccard_score)) 
         #13. resim
        image_name = results2[12][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img13.size())
        self.img13.setScene(scene2)        
        jaccard_score =str(results2[12][4])
        self.jacdeger13.setText(str(jaccard_score)) 
         #14. resim
        image_name = results2[13][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img14.size())
        self.img14.setScene(scene2)        
        jaccard_score =str(results2[13][4])
        self.jacdeger14.setText(str(jaccard_score)) 
         #15. resim
        image_name = results2[14][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img15.size())
        self.img15.setScene(scene2)        
        jaccard_score =str(results2[14][4])
        self.jacdeger15.setText(str(jaccard_score)) 
         #16. resim
        image_name = results2[15][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img16.size())
        self.img16.setScene(scene2)        
        jaccard_score =str(results2[15][4])
        self.jacdeger16.setText(str(jaccard_score)) 
         #17. resim
        image_name = results2[16][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img17.size())
        self.img17.setScene(scene2)        
        jaccard_score =str(results2[16][4])
        self.jacdeger17.setText(str(jaccard_score)) 
         #18. resim
        image_name = results2[17][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img18.size())
        self.img18.setScene(scene2)        
        jaccard_score =str(results2[17][4])
        self.jacdeger18.setText(str(jaccard_score)) 
         #19. resim
        image_name = results2[18][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img19.size())
        self.img19.setScene(scene2)        
        jaccard_score =str(results2[18][4])
        self.jacdeger19.setText(str(jaccard_score)) 
         #20. resim
        image_name = results2[19][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img20.size())
        self.img20.setScene(scene2)        
        jaccard_score =str(results2[19][4])
        self.jacdeger20.setText(str(jaccard_score)) 
         #21. resim
        image_name = results2[20][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img21.size())
        self.img21.setScene(scene2)        
        jaccard_score =str(results2[20][4])
        self.jacdeger21.setText(str(jaccard_score)) 
         #22. resim
        image_name = results2[21][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img22.size())
        self.img22.setScene(scene2)        
        jaccard_score =str(results2[21][4])
        self.jacdeger22.setText(str(jaccard_score)) 
         #23. resim
        image_name = results2[22][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img23.size())
        self.img23.setScene(scene2)        
        jaccard_score =str(results2[22][4])
        self.jacdeger23.setText(str(jaccard_score)) 
         #24. resim
        image_name = results2[23][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img24.size())
        self.img24.setScene(scene2)        
        jaccard_score =str(results2[23][4])
        self.jacdeger24.setText(str(jaccard_score)) 
         #25. resim
        image_name = results2[24][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img25.size())
        self.img25.setScene(scene2)        
        jaccard_score =str(results2[24][4])
        self.jacdeger25.setText(str(jaccard_score)) 
         #26. resim
        image_name = results2[25][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img26.size())
        self.img26.setScene(scene2)        
        jaccard_score =str(results2[25][4])
        self.jacdeger26.setText(str(jaccard_score)) 
         #27. resim
        image_name = results2[26][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img27.size())
        self.img27.setScene(scene2)        
        jaccard_score =str(results2[26][4])
        self.jacdeger27.setText(str(jaccard_score)) 
         #28. resim
        image_name = results2[27][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img28.size())
        self.img28.setScene(scene2)        
        jaccard_score =str(results2[27][4])
        self.jacdeger28.setText(str(jaccard_score)) 
         #29. resim
        image_name = results2[28][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img29.size())
        self.img29.setScene(scene2)        
        jaccard_score =str(results2[28][4])
        self.jacdeger29.setText(str(jaccard_score)) 
         #30. resim
        image_name = results2[29][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img30.size())
        self.img30.setScene(scene2)        
        jaccard_score =str(results2[29][4])
        self.jacdeger30.setText(str(jaccard_score))
         #31. resim
        image_name = results2[30][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img31.size())
        self.img31.setScene(scene2)        
        jaccard_score =str(results2[3][4])
        self.jacdeger31.setText(str(jaccard_score)) 
         #32. resim
        image_name = results2[31][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img32.size())
        self.img32.setScene(scene2)        
        jaccard_score =str(results2[31][4])
        self.jacdeger32.setText(str(jaccard_score)) 
         #33. resim
        image_name = results2[32][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img33.size())
        self.img33.setScene(scene2)        
        jaccard_score =str(results2[32][4])
        self.jacdeger33.setText(str(jaccard_score)) 
         #34. resim
        image_name = results2[33][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img34.size())
        self.img34.setScene(scene2)        
        jaccard_score =str(results2[33][4])
        self.jacdeger34.setText(str(jaccard_score)) 
         #35. resim
        image_name = results2[34][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img35.size())
        self.img35.setScene(scene2)        
        jaccard_score =str(results2[34][4])
        self.jacdeger35.setText(str(jaccard_score)) 
         #36. resim
        image_name = results2[35][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img36.size())
        self.img36.setScene(scene2)        
        jaccard_score =str(results2[35][4])
        self.jacdeger36.setText(str(jaccard_score)) 
         #37. resim
        image_name = results2[36][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img37.size())
        self.img37.setScene(scene2)        
        jaccard_score =str(results2[36][4])
        self.jacdeger37.setText(str(jaccard_score)) 
         #38. resim
        image_name = results2[37][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img38.size())
        self.img38.setScene(scene2)        
        jaccard_score =str(results2[37][4])
        self.jacdeger38.setText(str(jaccard_score)) 
         #39. resim
        image_name = results2[38][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img39.size())
        self.img39.setScene(scene2)        
        jaccard_score =str(results2[38][4])
        self.jacdeger39.setText(str(jaccard_score)) 
         #40. resim
        image_name = results2[39][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img40.size())
        self.img40.setScene(scene2)        
        jaccard_score =str(results2[39][4])
        self.jacdeger40.setText(str(jaccard_score)) 
        
    def siftuygula(self):        
        self.table_keypoint_result_2.clear()       
        results2=[]
        count_point2=0        
        dec2 = cv2.xfeatures2d.SIFT_create()
        for i,value in enumerate(self.dosyadataset):
            deger_img_source = cv2.imread(self.directory_retina_source+value[0])
            img_referance = cv2.imread(self.directory_retina_source+value[0],cv2.COLOR_BGR2GRAY)
            img_referance = self.resimkes(img_referance,value[2],value[1],value[3])
            img_source = cv2.imread(self.directory_retina_source+value[0],cv2.COLOR_BGR2GRAY)

            img_referance = self.step_I(img_referance,None)
            img_source = self.step_I(img_source,self.directory_retina_clahe+value[0])
            keypoints, descriptors = dec2.detectAndCompute(img_source, None)
            best_score = 0
            jaccard_score = 0
            gecicix,geciciy,gecicir = self.dosyadataset[i][2],self.dosyadataset[i][1],self.dosyadataset[i][3]
            deger_name,deger_x,deger_y,deger_r='',0,0,0
            deger_img = 0
            for index in range(len(keypoints)):
                count_point2 += 1
                x = int(keypoints[index].pt[0])
                y = int(keypoints[index].pt[1])
                r = value[3]                            
                img_source_point = self.resimkes(img_source,x,y,r)               
                if(img_referance.shape == img_source_point.shape):
                    score = self.ssim2(img_referance,img_source_point)#jaccard_similarity_score
                    if(score>best_score):
                        best_score=score
                        jaccard_score= self.jaccard(img_referance,img_source_point)
                        deger_img = img_source#img_source_point
                        deger_name = value[0]
                        deger_x,deger_y,deger_r=x,y,r            
            results2.append([deger_name,deger_y,deger_x,deger_r,best_score,jaccard_score])
            deger_img=cv2.rectangle(deger_img_source,(gecicix-gecicir,geciciy-gecicir),(gecicix+gecicir,geciciy+gecicir),(255,0,0),3)            
            deger_img=cv2.rectangle(deger_img,(deger_x-deger_r,deger_y-deger_r),(deger_x+deger_r,deger_y+deger_r),(255,0,0),3)
            cv2.imwrite(self.directory_retina_result+deger_name,deger_img)
        for i,row in enumerate(results2):
            for j, value in enumerate(row):
                self.table_keypoint_result_3.setItem(i,j, QTableWidgetItem(str(value)))        
        self.table_keypoint_result_3.horizontalHeader().setStretchLastSection(True)
        self.table_keypoint_result_3.resizeColumnsToContents()
        self.table_keypoint_result_3.setHorizontalHeaderLabels(self.label_header_keypoint)
           
        
        #1. resim
        image_name = results2[0][0]
        scene = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img1.size())
        self.img1.setScene(scene)        
        jaccard_score =str(results2[0][4])
        self.jacdeger1.setText(str(jaccard_score))
         #2. resim
        image_name = results2[1][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img2.size())
        self.img2.setScene(scene2)        
        jaccard_score =str(results2[1][4])
        self.jacdeger2.setText(str(jaccard_score))           
         #3. resim
        image_name = results2[2][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img3.size())
        self.img3.setScene(scene2)        
        jaccard_score =str(results2[2][4])
        self.jacdeger3.setText(str(jaccard_score)) 
         #4. resim
        image_name = results2[3][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img4.size())
        self.img4.setScene(scene2)        
        jaccard_score =str(results2[3][4])
        self.jacdeger4.setText(str(jaccard_score))         
         #5. resim
        image_name = results2[4][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img5.size())
        self.img5.setScene(scene2)        
        jaccard_score =str(results2[4][4])
        self.jacdeger5.setText(str(jaccard_score)) 
         #6. resim
        image_name = results2[5][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img6.size())
        self.img6.setScene(scene2)        
        jaccard_score =str(results2[5][4])
        self.jacdeger6.setText(str(jaccard_score)) 
         #7. resim
        image_name = results2[6][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img7.size())
        self.img7.setScene(scene2)        
        jaccard_score =str(results2[6][4])
        self.jacdeger7.setText(str(jaccard_score)) 
         #8. resim
        image_name = results2[7][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img8.size())
        self.img8.setScene(scene2)        
        jaccard_score =str(results2[7][4])
        self.jacdeger8.setText(str(jaccard_score)) 
         #9. resim
        image_name = results2[9][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img9.size())
        self.img9.setScene(scene2)        
        jaccard_score =str(results2[9][4])
        self.jacdeger9.setText(str(jaccard_score)) 
         #10. resim
        image_name = results2[9][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img10.size())
        self.img10.setScene(scene2)        
        jaccard_score =str(results2[9][4])
        self.jacdeger10.setText(str(jaccard_score)) 
        
         #11. resim
        image_name = results2[10][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img11.size())
        self.img11.setScene(scene2)        
        jaccard_score =str(results2[10][4])
        self.jacdeger11.setText(str(jaccard_score)) 
         #12. resim
        image_name = results2[11][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img12.size())
        self.img12.setScene(scene2)        
        jaccard_score =str(results2[11][4])
        self.jacdeger12.setText(str(jaccard_score)) 
         #13. resim
        image_name = results2[12][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img13.size())
        self.img13.setScene(scene2)        
        jaccard_score =str(results2[12][4])
        self.jacdeger13.setText(str(jaccard_score)) 
         #14. resim
        image_name = results2[13][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img14.size())
        self.img14.setScene(scene2)        
        jaccard_score =str(results2[13][4])
        self.jacdeger14.setText(str(jaccard_score)) 
         #15. resim
        image_name = results2[14][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img15.size())
        self.img15.setScene(scene2)        
        jaccard_score =str(results2[14][4])
        self.jacdeger15.setText(str(jaccard_score)) 
         #16. resim
        image_name = results2[15][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img16.size())
        self.img16.setScene(scene2)        
        jaccard_score =str(results2[15][4])
        self.jacdeger16.setText(str(jaccard_score)) 
         #17. resim
        image_name = results2[16][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img17.size())
        self.img17.setScene(scene2)        
        jaccard_score =str(results2[16][4])
        self.jacdeger17.setText(str(jaccard_score)) 
         #18. resim
        image_name = results2[17][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img18.size())
        self.img18.setScene(scene2)        
        jaccard_score =str(results2[17][4])
        self.jacdeger18.setText(str(jaccard_score)) 
         #19. resim
        image_name = results2[18][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img19.size())
        self.img19.setScene(scene2)        
        jaccard_score =str(results2[18][4])
        self.jacdeger19.setText(str(jaccard_score)) 
         #20. resim
        image_name = results2[19][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img20.size())
        self.img20.setScene(scene2)        
        jaccard_score =str(results2[19][4])
        self.jacdeger20.setText(str(jaccard_score)) 
         #21. resim
        image_name = results2[20][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img21.size())
        self.img21.setScene(scene2)        
        jaccard_score =str(results2[20][4])
        self.jacdeger21.setText(str(jaccard_score)) 
         #22. resim
        image_name = results2[21][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img22.size())
        self.img22.setScene(scene2)        
        jaccard_score =str(results2[21][4])
        self.jacdeger22.setText(str(jaccard_score)) 
         #23. resim
        image_name = results2[22][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img23.size())
        self.img23.setScene(scene2)        
        jaccard_score =str(results2[22][4])
        self.jacdeger23.setText(str(jaccard_score)) 
         #24. resim
        image_name = results2[23][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img24.size())
        self.img24.setScene(scene2)        
        jaccard_score =str(results2[23][4])
        self.jacdeger24.setText(str(jaccard_score)) 
         #25. resim
        image_name = results2[24][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img25.size())
        self.img25.setScene(scene2)        
        jaccard_score =str(results2[24][4])
        self.jacdeger25.setText(str(jaccard_score)) 
         #26. resim
        image_name = results2[25][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img26.size())
        self.img26.setScene(scene2)        
        jaccard_score =str(results2[25][4])
        self.jacdeger26.setText(str(jaccard_score)) 
         #27. resim
        image_name = results2[26][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img27.size())
        self.img27.setScene(scene2)        
        jaccard_score =str(results2[26][4])
        self.jacdeger27.setText(str(jaccard_score)) 
         #28. resim
        image_name = results2[27][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img28.size())
        self.img28.setScene(scene2)        
        jaccard_score =str(results2[27][4])
        self.jacdeger28.setText(str(jaccard_score)) 
         #29. resim
        image_name = results2[28][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img29.size())
        self.img29.setScene(scene2)        
        jaccard_score =str(results2[28][4])
        self.jacdeger29.setText(str(jaccard_score)) 
         #30. resim
        image_name = results2[29][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img30.size())
        self.img30.setScene(scene2)        
        jaccard_score =str(results2[29][4])
        self.jacdeger30.setText(str(jaccard_score))
         #31. resim
        image_name = results2[30][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img31.size())
        self.img31.setScene(scene2)        
        jaccard_score =str(results2[3][4])
        self.jacdeger31.setText(str(jaccard_score)) 
         #32. resim
        image_name = results2[31][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img32.size())
        self.img32.setScene(scene2)        
        jaccard_score =str(results2[31][4])
        self.jacdeger32.setText(str(jaccard_score)) 
         #33. resim
        image_name = results2[32][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img33.size())
        self.img33.setScene(scene2)        
        jaccard_score =str(results2[32][4])
        self.jacdeger33.setText(str(jaccard_score)) 
         #34. resim
        image_name = results2[33][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img34.size())
        self.img34.setScene(scene2)        
        jaccard_score =str(results2[33][4])
        self.jacdeger34.setText(str(jaccard_score)) 
         #35. resim
        image_name = results2[34][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img35.size())
        self.img35.setScene(scene2)        
        jaccard_score =str(results2[34][4])
        self.jacdeger35.setText(str(jaccard_score)) 
         #36. resim
        image_name = results2[35][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img36.size())
        self.img36.setScene(scene2)        
        jaccard_score =str(results2[35][4])
        self.jacdeger36.setText(str(jaccard_score)) 
         #37. resim
        image_name = results2[36][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img37.size())
        self.img37.setScene(scene2)        
        jaccard_score =str(results2[36][4])
        self.jacdeger37.setText(str(jaccard_score)) 
         #38. resim
        image_name = results2[37][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img38.size())
        self.img38.setScene(scene2)        
        jaccard_score =str(results2[37][4])
        self.jacdeger38.setText(str(jaccard_score)) 
         #39. resim
        image_name = results2[38][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img39.size())
        self.img39.setScene(scene2)        
        jaccard_score =str(results2[38][4])
        self.jacdeger39.setText(str(jaccard_score)) 
         #40. resim
        image_name = results2[39][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img40.size())
        self.img40.setScene(scene2)        
        jaccard_score =str(results2[39][4])
        self.jacdeger40.setText(str(jaccard_score))  
        
    def SURF_UYGULA(self):       
        self.table_keypoint_result.clear()      
        results2=[]
        count_point=0        
        dec = cv2.xfeatures2d.SURF_create()
        for i,value in enumerate(self.dosyadataset):
            deger_img_source = cv2.imread(self.directory_retina_source+value[0])
            img_referance = cv2.imread(self.directory_retina_source+value[0],cv2.COLOR_BGR2GRAY)
            img_referance = self.resimkes(img_referance,value[2],value[1],value[3])
            img_source = cv2.imread(self.directory_retina_source+value[0],cv2.COLOR_BGR2GRAY)
            img_referance = self.step_I(img_referance,None)
            img_source = self.step_I(img_source,self.directory_retina_clahe+value[0])
            keypoints, descriptors = dec.detectAndCompute(img_source, None)
            best_score = 0
            jaccard_score = 0
            gecicix,geciciy,gecicir = self.dosyadataset[i][2],self.dosyadataset[i][1],self.dosyadataset[i][3]
            deger_name,deger_x,deger_y,deger_r='',0,0,0
            deger_img = 0
            for index in range(len(keypoints)):
                count_point += 1
                x = int(keypoints[index].pt[0])
                y = int(keypoints[index].pt[1])
                r = value[3]                            
                img_source_point = self.resimkes(img_source,x,y,r)               
                if(img_referance.shape == img_source_point.shape):
                    score = self.ssim2(img_referance,img_source_point)#jaccard_similarity_score
                    if(score>best_score):
                        best_score=score
                        jaccard_score= self.jaccard(img_referance,img_source_point)
                        deger_img = img_source#img_source_point
                        deger_name = value[0]
                        deger_x,deger_y,deger_r=x,y,r            
            results2.append([deger_name,deger_y,deger_x,deger_r,best_score,jaccard_score])
            deger_img=cv2.rectangle(deger_img_source,(gecicix-gecicir,geciciy-gecicir),(gecicix+gecicir,geciciy+gecicir),(255,0,0),3)            
            deger_img=cv2.rectangle(deger_img,(deger_x-deger_r,deger_y-deger_r),(deger_x+deger_r,deger_y+deger_r),(255,0,0),3)
            cv2.imwrite(self.directory_retina_result+deger_name,deger_img)
    
        self.table_keypoint_result.setColumnCount(len(results2[0]))
        self.table_keypoint_result.setRowCount(len(results2))
        for i,row in enumerate(results2):
            for j, value in enumerate(row):
                self.table_keypoint_result.setItem(i,j, QTableWidgetItem(str(value)))
        
        self.table_keypoint_result.horizontalHeader().setStretchLastSection(True)
        self.table_keypoint_result.resizeColumnsToContents()
        self.table_keypoint_result.setHorizontalHeaderLabels(self.label_header_keypoint)
        
        
        #1. resim
        image_name = results2[0][0]
        scene = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img1.size())
        self.img1.setScene(scene)        
        jaccard_score =str(results2[0][4])
        self.jacdeger1.setText(str(jaccard_score))
         #2. resim
        image_name = results2[1][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img2.size())
        self.img2.setScene(scene2)        
        jaccard_score =str(results2[1][4])
        self.jacdeger2.setText(str(jaccard_score))           
         #3. resim
        image_name = results2[2][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img3.size())
        self.img3.setScene(scene2)        
        jaccard_score =str(results2[2][4])
        self.jacdeger3.setText(str(jaccard_score)) 
         #4. resim
        image_name = results2[3][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img4.size())
        self.img4.setScene(scene2)        
        jaccard_score =str(results2[3][4])
        self.jacdeger4.setText(str(jaccard_score))         
         #5. resim
        image_name = results2[4][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img5.size())
        self.img5.setScene(scene2)        
        jaccard_score =str(results2[4][4])
        self.jacdeger5.setText(str(jaccard_score)) 
         #6. resim
        image_name = results2[5][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img6.size())
        self.img6.setScene(scene2)        
        jaccard_score =str(results2[5][4])
        self.jacdeger6.setText(str(jaccard_score)) 
         #7. resim
        image_name = results2[6][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img7.size())
        self.img7.setScene(scene2)        
        jaccard_score =str(results2[6][4])
        self.jacdeger7.setText(str(jaccard_score)) 
         #8. resim
        image_name = results2[7][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img8.size())
        self.img8.setScene(scene2)        
        jaccard_score =str(results2[7][4])
        self.jacdeger8.setText(str(jaccard_score)) 
         #9. resim
        image_name = results2[9][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img9.size())
        self.img9.setScene(scene2)        
        jaccard_score =str(results2[9][4])
        self.jacdeger9.setText(str(jaccard_score)) 
         #10. resim
        image_name = results2[9][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img10.size())
        self.img10.setScene(scene2)        
        jaccard_score =str(results2[9][4])
        self.jacdeger10.setText(str(jaccard_score)) 
        
         #11. resim
        image_name = results2[10][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img11.size())
        self.img11.setScene(scene2)        
        jaccard_score =str(results2[10][4])
        self.jacdeger11.setText(str(jaccard_score)) 
         #12. resim
        image_name = results2[11][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img12.size())
        self.img12.setScene(scene2)        
        jaccard_score =str(results2[11][4])
        self.jacdeger12.setText(str(jaccard_score)) 
         #13. resim
        image_name = results2[12][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img13.size())
        self.img13.setScene(scene2)        
        jaccard_score =str(results2[12][4])
        self.jacdeger13.setText(str(jaccard_score)) 
         #14. resim
        image_name = results2[13][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img14.size())
        self.img14.setScene(scene2)        
        jaccard_score =str(results2[13][4])
        self.jacdeger14.setText(str(jaccard_score)) 
         #15. resim
        image_name = results2[14][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img15.size())
        self.img15.setScene(scene2)        
        jaccard_score =str(results2[14][4])
        self.jacdeger15.setText(str(jaccard_score)) 
         #16. resim
        image_name = results2[15][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img16.size())
        self.img16.setScene(scene2)        
        jaccard_score =str(results2[15][4])
        self.jacdeger16.setText(str(jaccard_score)) 
         #17. resim
        image_name = results2[16][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img17.size())
        self.img17.setScene(scene2)        
        jaccard_score =str(results2[16][4])
        self.jacdeger17.setText(str(jaccard_score)) 
         #18. resim
        image_name = results2[17][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img18.size())
        self.img18.setScene(scene2)        
        jaccard_score =str(results2[17][4])
        self.jacdeger18.setText(str(jaccard_score)) 
         #19. resim
        image_name = results2[18][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img19.size())
        self.img19.setScene(scene2)        
        jaccard_score =str(results2[18][4])
        self.jacdeger19.setText(str(jaccard_score)) 
         #20. resim
        image_name = results2[19][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img20.size())
        self.img20.setScene(scene2)        
        jaccard_score =str(results2[19][4])
        self.jacdeger20.setText(str(jaccard_score)) 
         #21. resim
        image_name = results2[20][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img21.size())
        self.img21.setScene(scene2)        
        jaccard_score =str(results2[20][4])
        self.jacdeger21.setText(str(jaccard_score)) 
         #22. resim
        image_name = results2[21][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img22.size())
        self.img22.setScene(scene2)        
        jaccard_score =str(results2[21][4])
        self.jacdeger22.setText(str(jaccard_score)) 
         #23. resim
        image_name = results2[22][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img23.size())
        self.img23.setScene(scene2)        
        jaccard_score =str(results2[22][4])
        self.jacdeger23.setText(str(jaccard_score)) 
         #24. resim
        image_name = results2[23][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img24.size())
        self.img24.setScene(scene2)        
        jaccard_score =str(results2[23][4])
        self.jacdeger24.setText(str(jaccard_score)) 
         #25. resim
        image_name = results2[24][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img25.size())
        self.img25.setScene(scene2)        
        jaccard_score =str(results2[24][4])
        self.jacdeger25.setText(str(jaccard_score)) 
         #26. resim
        image_name = results2[25][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img26.size())
        self.img26.setScene(scene2)        
        jaccard_score =str(results2[25][4])
        self.jacdeger26.setText(str(jaccard_score)) 
         #27. resim
        image_name = results2[26][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img27.size())
        self.img27.setScene(scene2)        
        jaccard_score =str(results2[26][4])
        self.jacdeger27.setText(str(jaccard_score)) 
         #28. resim
        image_name = results2[27][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img28.size())
        self.img28.setScene(scene2)        
        jaccard_score =str(results2[27][4])
        self.jacdeger28.setText(str(jaccard_score)) 
         #29. resim
        image_name = results2[28][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img29.size())
        self.img29.setScene(scene2)        
        jaccard_score =str(results2[28][4])
        self.jacdeger29.setText(str(jaccard_score)) 
         #30. resim
        image_name = results2[29][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img30.size())
        self.img30.setScene(scene2)        
        jaccard_score =str(results2[29][4])
        self.jacdeger30.setText(str(jaccard_score))
         #31. resim
        image_name = results2[30][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img31.size())
        self.img31.setScene(scene2)        
        jaccard_score =str(results2[3][4])
        self.jacdeger31.setText(str(jaccard_score)) 
         #32. resim
        image_name = results2[31][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img32.size())
        self.img32.setScene(scene2)        
        jaccard_score =str(results2[31][4])
        self.jacdeger32.setText(str(jaccard_score)) 
         #33. resim
        image_name = results2[32][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img33.size())
        self.img33.setScene(scene2)        
        jaccard_score =str(results2[32][4])
        self.jacdeger33.setText(str(jaccard_score)) 
         #34. resim
        image_name = results2[33][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img34.size())
        self.img34.setScene(scene2)        
        jaccard_score =str(results2[33][4])
        self.jacdeger34.setText(str(jaccard_score)) 
         #35. resim
        image_name = results2[34][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img35.size())
        self.img35.setScene(scene2)        
        jaccard_score =str(results2[34][4])
        self.jacdeger35.setText(str(jaccard_score)) 
         #36. resim
        image_name = results2[35][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img36.size())
        self.img36.setScene(scene2)        
        jaccard_score =str(results2[35][4])
        self.jacdeger36.setText(str(jaccard_score)) 
         #37. resim
        image_name = results2[36][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img37.size())
        self.img37.setScene(scene2)        
        jaccard_score =str(results2[36][4])
        self.jacdeger37.setText(str(jaccard_score)) 
         #38. resim
        image_name = results2[37][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img38.size())
        self.img38.setScene(scene2)        
        jaccard_score =str(results2[37][4])
        self.jacdeger38.setText(str(jaccard_score)) 
         #39. resim
        image_name = results2[38][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img39.size())
        self.img39.setScene(scene2)        
        jaccard_score =str(results2[38][4])
        self.jacdeger39.setText(str(jaccard_score)) 
         #40. resim
        image_name = results2[39][0]
        scene2 = self.show_image_path(self.directory_retina_result+"/"+image_name,self.img40.size())
        self.img40.setScene(scene2)        
        jaccard_score =str(results2[39][4])
        self.jacdeger40.setText(str(jaccard_score))        
    
    def step_II(self,img,path):
        teval,img = cv2.threshold(img,10,255, cv2.THRESH_BINARY)
        if(path != None):
            cv2.imwrite(path,img)
        return img
    
    def step_I(self,img,path):
        #img = color.rgb2gray(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        img = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img[0])
        if(path != None):
            cv2.imwrite(path,img)
        return img
    
    def step_III(self,img,path):
        img = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        if(path != None):
            cv2.imwrite(path,img)
        return img
    
    def resimkes(self,img,x,y,r):
        w1,h1=x-r,y-r
        w2,h2 = x+r,y+r
        img = img[h1:h2, w1:w2]
        return img
    

    
    def key_show(self,img,key):
        img = cv2.drawKeypoints(img, key, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #metodlar
    def show_image_path(self,img_path,size):
        self.pixmap = Qt.QPixmap()
        self.pixmap.load(img_path)
        self.pixmap = self.pixmap.scaled(size, Qt.Qt.KeepAspectRatioByExpanding,transformMode=QtCore.Qt.SmoothTransformation)
        self.graphicsPixmapItem = Qt.QGraphicsPixmapItem(self.pixmap)
        self.graphicsScene = override_graphicsScene(self)
        self.graphicsScene.addItem(self.graphicsPixmapItem)
        return self.graphicsScene
    
    def ssim(self,img1,img2):
        img_1 = np.asarray(img1)#cv2.imread(img1)
        img_2 = np.asarray(img2)#cv2.imread(img2)

        img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
        img_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
        try:
            if(not img_1 is None and not img_2 is None):
                if(img_1.size == img_2.size):
                    return round(SSIM2(img_1,img_2),2)
            else:
                return 0.0
        except ValueError:
            print("Invalid Entry - try again")
            return 0.0
        return 0.0
    
    def mse(self,img1,img2):
        img_1 = cv2.imread(img1)
        img_2 = cv2.imread(img2)
        
        e = np.sum((img_1.astype("float") - img_2.astype("float"))**2)
        e /= float(img_1.shape[0] * img_2.shape[1])
        r = round(e,2)
        return r
    
    def jaccard(self,img1,img2):
        img_true=np.array(img1).ravel()
        img_pred=np.array(img2).ravel()
        iou = jaccard_similarity_score(img_true, img_pred)
        return iou
    
    def ssim2(self,img1,img2):
        img_1 = np.asarray(img1)#cv2.imread(img1)
        img_2 = np.asarray(img2)#cv2.imread(img2)

        """img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
        img_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)"""
        try:
            if(not img_1 is None and not img_2 is None):
                if(img_1.size == img_2.size):
                    return round(SSIM2(img_1,img_2),2)
            else:
                return 0.0
        except ValueError:
            print("Invalid Entry - try again")
            return 0.0
        return 0.0
    
    
    def mse2(self,img1,img2):
        img_1 = img1
        img_2 = img2
        
        e = np.sum((img_1.astype("float") - img_2.astype("float"))**2)
        e /= float(img_1.shape[0] * img_2.shape[1])
        r = round(e,2)
        return r
    
    def dosyakaydet(self,list_):
        with open('dataset.txt', 'w') as f:
            for item in list_:
                f.write("%s\n" % item)
    
    def show_img(self,img):
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()