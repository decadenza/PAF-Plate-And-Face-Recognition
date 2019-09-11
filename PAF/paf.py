#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
PAF - Plate and Faces near recognition
Pasquale Lafiosca (c) 2019
"""

import sys, os, time, datetime
from PyQt5 import QtCore, QtWidgets, uic
from GUI import resources
import json
import csv
import dlib
import cv2
import numpy as np
import sqlite3 as sql
from threading import Thread
import multiprocessing as mp
import queue

import settings # Local settings
from lib.capture import Capture
import openalpr

# Global constants

FACE_DETECTOR = dlib.get_frontal_face_detector() # Create a HOG face detector using the built-in dlib class
FACE_POSE_PREDICTOR = dlib.shape_predictor(settings.SHAPE_PREDICTOR) # Getting landmarks
FACE_RECOGNITION_MODEL = dlib.face_recognition_model_v1(settings.FACE_RECOGNITION_MODEL) # Getting 128 measures
ALPR = openalpr.Alpr(settings.OPENALPR_COUNTRY, settings.OPENALPR_CONF, settings.OPENALPR_RUNTIME_DATA)
if not ALPR.is_loaded():
    raise ImportError("Error loading OpenALPR library.")
ALPR.set_top_n(1) # Get only best result        
    
class mainWindow(QtWidgets.QMainWindow):
    
    errorInWorker = QtCore.pyqtSignal(Exception)
    
    def __init__(self):
        super().__init__() #inheriting from the object.
        uic.loadUi(os.path.join(settings.CUR_PATH,'GUI','main.ui'), self)
        # Menu connections
        self.actionAbout.triggered.connect(self.openAbout)
        self.actionAnalyze.triggered.connect(self.openAnalyze)
        self.actionTargetManager.triggered.connect(lambda: self.setCurrentWidget(targetManager(self)))
        self.actionHome.triggered.connect(lambda: self.setCurrentWidget(home(self)))
        self.actionExit.triggered.connect(self.close)
        # Status bar
        now = datetime.datetime.now()
        self.statusCurTime = QtWidgets.QLabel(now.strftime("%d/%m/%Y %H:%M:%S"))
        self.statusCurTime.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.statusBar.addWidget(self.statusCurTime, 1)
        self.statusInfo = QtWidgets.QLabel("")
        self.statusBar.addWidget(self.statusInfo, 2)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateCurTime)
        self.timer.start(1000)
        self.setCurrentWidget(home(self)) # Show main widget as child after startup
        centerOnScreen(self) # Centers window on the screen
        self.CvWindowIsOpen = False
        self.mainPool = None
        self.errorInWorker.connect(self.workerError)
        self.asyncResults = []
        
    # Open widget in current window and add padding
    def setCurrentWidget(self, w):
        self.setCentralWidget(w)
        self.setFixedSize(w.width(), w.height())
    
    def openAbout(self):
        self.about = about(parent=self)
        self.about.show()
    
    def openAnalyze(self):
        self.analyze = analyzeFile(parent=self)
        self.analyze.show()
    
    def updateCurTime(self):
        self.statusCurTime.setText(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                  
    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, 'Confirmation', "Do you REALLY want to exit and suspend all the background processes?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.CvWindowIsOpen = False # Force closing Cv2 Window
            if self.mainPool:
                self.mainPool.terminate()
            event.accept()
        else:
            event.ignore()
    
    def recognitionInitialization(self):
        self.setCursor(QtCore.Qt.WaitCursor)
        # Load cameras data
        cams = DB.execute("SELECT id, name, activeFace, activePlate FROM cameras ORDER BY id LIMIT 4").fetchall() # 4 cameras hardcoded
        faceProcesses = 0
        plateProcesses = 0
        backgroundProcesses = 0
        for c in cams:
            if c[2] or c[3]:
                backgroundProcesses+=1
            if c[2]:
                faceProcesses+=1 # Statistics only
            if c[3]:
                plateProcesses+=1
            
        #Pool initialization
        self.mainPool = mp.Pool(processes=max(backgroundProcesses,1)) # start worker processes for faces and plates
        for i, c in enumerate(cams):
            if c[2] or c[3]: # Unique recognition process
                self.asyncResults.append( self.mainPool.apply_async(recognitionProcess, args=(c[0], ), error_callback=self.errorInWorker.emit) )
            
        self.mainPool.close() # No more tasks can be added
        self.statusInfo.setText("Running processes: %s Face recognition, %s Plate recognition" % (faceProcesses, plateProcesses))
        self.unsetCursor()
        
        
        
    def reInitializeProcesses(self): # RESTART
        if self.mainPool:
            self.mainPool.terminate()
        self.recognitionInitialization()
    
    def workerError(self, e): # CRITICAL ERROR (not managed) IN WORKER.
        self.statusInfo.setText("Unknown error. Please restart the application.")
        QtWidgets.QMessageBox(parent = self, icon = QtWidgets.QMessageBox.Critical,
                    windowTitle="Unknown error!",
                    text="A unknown error has happened in one of the background processes. Please check your settings and video sources and restart the application.",
                    standardButtons=QtWidgets.QMessageBox.Ok,
                    detailedText=repr(e)
                    ).exec_()
                              
# Widgets
class home(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        uic.loadUi(os.path.join(settings.CUR_PATH,'GUI','home.ui'), self)
        # Load cameras data
        cams = DB.execute("SELECT id, name, activeFace, activePlate FROM cameras ORDER BY id LIMIT 4").fetchall() # 4 cameras hardcoded
        # Setup loading
        for i, r in enumerate(cams): # Button connections 
            getattr(self, "activateFace%s" % (i+1)).setChecked(r[2])
            getattr(self, "activatePlate%s" % (i+1)).setChecked(r[3])
            getattr(self, "configCamera%s" % (i+1)).clicked.connect(lambda arg, i=i, r=r: self.openConfigCamera(i, r[0], r[1]))
            getattr(self, "eventButton%s" % (i+1)).clicked.connect(lambda arg, i=i, r=r: self.openEventCamera(i, r[0], r[1]))
            getattr(self, "activateFace%s" % (i+1)).stateChanged.connect(lambda arg, i=i, r=r: self.changeStateFace(i, r[0])) # stateChanged returns a checked argument (always 0 or 2)
            getattr(self, "activatePlate%s" % (i+1)).stateChanged.connect(lambda arg, i=i, r=r: self.changeStatePlate(i, r[0]))
            getattr(self, "buttonCamera%s" % (i+1)).clicked.connect(lambda arg, r=r: self.openLive(r[0]))
            getattr(self, "positionLabel%s" % (i+1)).setText(r[1])
        
        
    def openConfigCamera(self, i, dbIndex, dbName):
        configCamera = configureCamera(self.parent, i, dbIndex, dbName)
        self.parent.setCurrentWidget(configCamera)
    
    def openEventCamera(self, i, dbIndex, dbName):
        e = eventCamera(self.parent, i, dbIndex, dbName)
        self.parent.setCurrentWidget(e)
    
    def openLive(self, dbIndex):
        if self.parent.CvWindowIsOpen:
            return
        self.setCursor(QtCore.Qt.WaitCursor)
        row = DB.execute("SELECT name, url, roi FROM cameras WHERE id = ? LIMIT 1", (dbIndex,)).fetchone()
        if row and row[1]:
            topLeft = None
            bottomRight = None
            if row[2]: # Show ROI
                r = [int(s) for s in row[2].split() if s.isdigit()]
                topLeft = (r[0], r[1])
                bottomRight =  (r[0]+r[2], r[1]+r[3])
            try:
                windowName = 'Live - %s' % row[0]
                cap = Capture(source=row[1])
                cap.start()
                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
                cv2.moveWindow(windowName,0,0)
                cv2.resizeWindow(windowName, 800,600)
                self.parent.CvWindowIsOpen = True
                self.unsetCursor()
                while(self.parent.CvWindowIsOpen):
                    frame = cap.get()
                    if frame is not None:
                        if topLeft:
                            cv2.rectangle(frame, topLeft, bottomRight, (255,0,0),3)
                        cv2.imshow(windowName, frame) # Display webcam
                    if cv2.waitKey(5) & 0xFF == 27: # If ESC is pressed
                        break
                cap.stop()
                cv2.destroyWindow(windowName)
                self.parent.CvWindowIsOpen = False
            except Exception as e:
                msg = QtWidgets.QMessageBox(parent = self, icon = QtWidgets.QMessageBox.Critical, windowTitle="Streaming failed!",
                    text="Impossible to open video streaming. Please check your URL configuration.", standardButtons=QtWidgets.QMessageBox.Ok)
                msg.setDetailedText(repr(e))
                msg.exec_()
                
        else:
            QtWidgets.QMessageBox.critical(self,
                                    "Streaming failed!",
                                    "Impossible to open video streaming. No URL given.",
                                    QtWidgets.QMessageBox.Ok,
                                    )
                                    
        
    def changeStateFace(self, i, dbIndex): # Sync gui and db
        self.setCursor(QtCore.Qt.WaitCursor)
        if getattr(self, "activateFace%s" % (i+1)).isChecked():
            DB.execute("UPDATE cameras SET activeFace = 1 WHERE id = ?", (dbIndex,) )
        else:
            DB.execute("UPDATE cameras SET activeFace = 0 WHERE id = ?", (dbIndex,) )
        self.parent.reInitializeProcesses()
        self.unsetCursor()
        
    def changeStatePlate(self, i, dbIndex): # Sync gui and db
        self.setCursor(QtCore.Qt.WaitCursor)
        if getattr(self, "activatePlate%s" % (i+1)).isChecked():
            DB.execute("UPDATE cameras SET activePlate = 1 WHERE id = ?", (dbIndex,) )
        else:
            DB.execute("UPDATE cameras SET activePlate = 0 WHERE id = ?", (dbIndex,) )        
        self.parent.reInitializeProcesses()
        self.unsetCursor()
    
    
class configureCamera(QtWidgets.QWidget):
    def __init__(self, parent, i, dbIndex, dbName=''):
        super().__init__() #inheriting from the object.
        self.parent = parent
        self.dbIndex = dbIndex
        uic.loadUi(os.path.join(settings.CUR_PATH,'GUI','configuracamera.ui'), self)
        self.title.setText('Camera configuration: %s' % dbName)
        self.buttonSave.clicked.connect(self.save)
        self.buttonBack.clicked.connect(self.goBack)
        self.selectROI.clicked.connect(self.openRoiSelection)
        self.unsetROI.clicked.connect(self.doUnsetRoi)
        row = DB.execute("SELECT name, url, saveNewFaces, saveNewPlates, roi FROM cameras WHERE id = ? LIMIT 1", (dbIndex,)).fetchone()
        self.name.setText(row[0])
        self.url.setPlainText(row[1])
        self.saveNewFace.setChecked(row[2])
        self.saveNewPlate.setChecked(row[3])
        self.roiValue.setText(row[4])
        
    def openRoiSelection(self):
        try:
            cap = Capture(source=self.url.toPlainText())
            cap.start()
            frame=None
            i=0
            while(frame is None and i<5):
                time.sleep(1)
                frame = cap.get()
                i+=1
            cap.stop()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self,
                                    "URL not found!",
                                    "Please check your internet connection and set a valid URL to point.",
                                    QtWidgets.QMessageBox.Ok
                                    )
            return
        else:
            if frame is None:
                QtWidgets.QMessageBox.warning(self,
                                        "No video!",
                                        "Please check your internet connection and set a valid URL to point.",
                                        QtWidgets.QMessageBox.Ok
                                        )
                return
            
            cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Select ROI',0,0)
            cv2.resizeWindow('Select ROI', 800,600)
            r = cv2.selectROI('Select ROI',frame,True,False)
            cv2.destroyWindow('Select ROI')
            if r and r[2] and r[3]:
                self.roiValue.setText( '{} {} {} {}'.format(r[0], r[1], r[2], r[3]) )
        
    def doUnsetRoi(self):
        self.roiValue.setText('')
    
    def goBack(self):
        back = home(self.parent)
        self.parent.setCurrentWidget(back)
    
    def save(self):
        do = True
        if not self.name.text():
            do = False
            QtWidgets.QMessageBox.warning(self,
                                    "Name not set!",
                                    "Please specify a name.",
                                    QtWidgets.QMessageBox.Ok
                                    )
        if not self.url.toPlainText():
            do = False
            QtWidgets.QMessageBox.warning(self,
                                    "URL not set!",
                                    "Please specify a URL.",
                                    QtWidgets.QMessageBox.Ok
                                    )
        if do:
            DB.execute("UPDATE cameras SET name = ?, url = ?, saveNewFaces=?, saveNewPlates=?, roi=? WHERE id = ?", (self.name.text(), self.url.toPlainText(), 1 if self.saveNewFace.isChecked() else 0, 1 if self.saveNewPlate.isChecked() else 0, self.roiValue.text(), self.dbIndex,) )
            self.parent.reInitializeProcesses()
            self.goBack()
    
    


class eventCamera(QtWidgets.QWidget):
    def __init__(self, parent, i, dbIndex, dbName=''):
        super().__init__() #inheriting from the object.
        self.parent = parent
        self.dbIndex = dbIndex
        uic.loadUi(os.path.join(settings.CUR_PATH,'GUI','eventicamera.ui'), self)
        self.title.setText('Events %s' % dbName)
        self.buttonBack.clicked.connect(self.goBack)
        self.buttonReset.clicked.connect(self.reset)
        self.savePath = os.path.join(settings.EVENTS_PATH, str(dbIndex))
        # Load table
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableWidget.setHorizontalHeaderLabels(['ID', 'Screenshot','Type','Target','Plate'])
        self.tableWidget.setColumnHidden(0, True) # Col 0 is hidden because contains only ID!
        self.loadEvents()
        self.tableWidget.sortItems(1, QtCore.Qt.DescendingOrder)
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.cellDoubleClicked.connect(self.openImage)
    
        
    def loadEvents(self):
        self.tableWidget.setRowCount(0)
        for i, row in enumerate( DB.execute("SELECT eventFaces.id, eventFaces.datetime, targetFaces.name FROM eventFaces LEFT JOIN targetFaces ON eventFaces.target = targetFaces.id WHERE eventFaces.camera = ?", (self.dbIndex,)) ):
            self.tableWidget.insertRow(i)
            self.tableWidget.setItem(i , 0, QtWidgets.QTableWidgetItem(str(row[0])))
            self.tableWidget.setItem(i , 1, QtWidgets.QTableWidgetItem(datetime.datetime.strptime(row[1],'%Y%m%d%H%M%S%f').strftime('%Y-%m-%d %H:%M:%S:%f')))    
            self.tableWidget.setItem(i , 2, QtWidgets.QTableWidgetItem('F'))
            self.tableWidget.setItem(i , 3, QtWidgets.QTableWidgetItem(row[2]))
        for i, row in enumerate( DB.execute("SELECT eventPlates.id, eventPlates.datetime, targetPlates.name, eventPlates.plate FROM eventPlates LEFT JOIN targetPlates ON eventPlates.target = targetPlates.id WHERE eventPlates.camera = ?", (self.dbIndex,)) ):
            self.tableWidget.insertRow(i)
            self.tableWidget.setItem(i , 0, QtWidgets.QTableWidgetItem(str(row[0])))
            self.tableWidget.setItem(i , 1, QtWidgets.QTableWidgetItem(datetime.datetime.strptime(row[1],'%Y%m%d%H%M%S%f').strftime('%Y-%m-%d %H:%M:%S:%f')))    
            self.tableWidget.setItem(i , 2, QtWidgets.QTableWidgetItem('P'))
            self.tableWidget.setItem(i , 3, QtWidgets.QTableWidgetItem(row[2]))
            self.tableWidget.setItem(i , 4, QtWidgets.QTableWidgetItem(row[3]))
    
    def openImage(self, row=None, column=None):
        if row is not None and column is not None:
            filename = self.tableWidget.item(row, 1).text().replace('-','').replace(':','').replace(' ','')+'.png'
            os.system('xdg-open "'+os.path.join(self.savePath,filename)+'"')
    
    def goBack(self):
        back = home(self.parent)
        self.parent.setCurrentWidget(back)
    
    def reset(self):
        reply = QtWidgets.QMessageBox.question(self, 'Confirmation', "Do you really want to DELETE all the events connected to this camera?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        savePath = os.path.join(settings.EVENTS_PATH, str(self.dbIndex))
        if reply == QtWidgets.QMessageBox.Yes:
            DB.execute("DELETE FROM eventFaces WHERE camera=?", (self.dbIndex, ) )
            DB.execute("DELETE FROM eventPlates WHERE camera=?", (self.dbIndex, ) )
            if os.path.isdir(savePath):
                for the_file in os.listdir(savePath):
                    file_path = os.path.join(savePath, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except:
                        pass
            self.loadEvents()
          
           
class targetManager(QtWidgets.QWidget):
    def __init__(self, parent):
        self.parent = parent
        super().__init__() #inheriting from the object.
        uic.loadUi(os.path.join(settings.CUR_PATH,'GUI','gestionetarget.ui'), self)
        #self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.buttonClose.clicked.connect(self.goBack)
        self.buttonNew.clicked.connect(self.openAddFace)
        self.buttonNew2.clicked.connect(self.openAddPlate)
        self.buttonDelete.clicked.connect(self.deleteFace)
        self.buttonDelete2.clicked.connect(self.deletePlate)
        #centerOnScreen(self) # Centers window on the screen
        # Set table Face
        self.targetFaceList.setColumnCount(2)
        self.targetFaceList.setHorizontalHeaderLabels(['ID', 'Target name'])
        self.targetFaceList.setColumnHidden(0, True) # Col 0 is hidden because contains only ID!
        self.targetFaceList.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.targetFaceList.cellDoubleClicked.connect(self.openAddFace)
        self.targetFaceList.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        # Set table Plate
        self.targetPlateList.setColumnCount(3)
        self.targetPlateList.setHorizontalHeaderLabels(['ID', 'Target name', 'Plate'])
        self.targetPlateList.setColumnHidden(0, True) # Col 0 is hidden because contains only ID!
        self.targetPlateList.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.targetPlateList.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.targetPlateList.cellDoubleClicked.connect(self.openAddPlate)
        self.targetPlateList.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        # Load data from db
        self.loadFaceTargets()
        self.loadPlateTargets()
        self.targetFaceList.sortItems(1, QtCore.Qt.AscendingOrder)
        self.targetPlateList.sortItems(1, QtCore.Qt.AscendingOrder)
        
    def loadFaceTargets(self):
        self.targetFaceList.setRowCount(0)
        for i, row in enumerate( DB.execute("SELECT id, name FROM targetFaces ORDER BY name") ):
            self.targetFaceList.insertRow(i)
            self.targetFaceList.setItem(i , 0, QtWidgets.QTableWidgetItem(str(row[0])))
            self.targetFaceList.setItem(i , 1, QtWidgets.QTableWidgetItem(row[1]))
    
    def loadPlateTargets(self):
        self.targetPlateList.setRowCount(0)
        for i, row in enumerate( DB.execute("SELECT id, name, plate FROM targetPlates ORDER BY name") ):
            self.targetPlateList.insertRow(i)
            self.targetPlateList.setItem(i , 0, QtWidgets.QTableWidgetItem(str(row[0])))
            self.targetPlateList.setItem(i , 1, QtWidgets.QTableWidgetItem(row[1]))
            self.targetPlateList.setItem(i , 2, QtWidgets.QTableWidgetItem(row[2]))
        
    def openAddFace(self, row=None, column=None):
        faceId = None
        if row is not None and column is not None:
            faceId = int(self.targetFaceList.item(row, 0).text())
        self.addFace = addFace(self, faceId)
        self.addFace.show()
    
    def openAddPlate(self, row=None, column=None):
        plateId = None
        if row is not None and column is not None:
            plateId = int(self.targetPlateList.item(row, 0).text())
        self.addPlate = addPlate(self, plateId)
        self.addPlate.show()
    
    def deletePlate(self):
        selectedRows = self.targetPlateList.selectionModel().selectedRows()
        if len(selectedRows) < 1:
            return
        reply = QtWidgets.QMessageBox.question(self, 'Confirmation', "Do you really want to DELETE selected plates?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            for e in selectedRows:
                plateId = int(self.targetPlateList.item(e.row(), 0).text())
                DB.execute("DELETE FROM targetPlates WHERE id=?", (plateId, ) )
                self.targetPlateList.removeRow(e.row())
            
    def deleteFace(self):
        selectedRows = self.targetFaceList.selectionModel().selectedRows()
        if len(selectedRows) < 1:
            return
        reply = QtWidgets.QMessageBox.question(self, 'Confirmation', "Do you really want to DELETE selected faces?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            for e in selectedRows:
                faceId = int(self.targetFaceList.item(e.row(), 0).text())
                DB.execute("DELETE FROM targetFaces WHERE id=?", (faceId, ) )
                self.targetFaceList.removeRow(e.row())    
        
    def goBack(self):
        back = home(self.parent)
        self.parent.setCurrentWidget(back)
        

class addPlate(QtWidgets.QDialog):
    def __init__(self, parent, plateId=None):
        super().__init__(parent) #inheriting from the object.
        uic.loadUi(os.path.join(settings.CUR_PATH,'GUI','aggiungitarga.ui'), self)
        #self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)gnome-desktop-item-edit --create-new ~/Desktop
        self.parent = parent
        self.saveButton.clicked.connect(self.save)
        self.curId = None
        # Load current face info
        if plateId is not None:
            res = DB.execute("SELECT name, plate FROM targetPlates WHERE id=(?) LIMIT 1", (plateId,)).fetchone()
            self.curId = plateId
            self.name.setText(res[0])
            self.plate.setText(res[1])
            
    
    def save(self):
        do = True
        if not self.name.text():
            do = False
            QtWidgets.QMessageBox.warning(self,
                                    "Name not set!",
                                    "Type a name to be used as reference to this target.",
                                    QtWidgets.QMessageBox.Ok
                                    )
        if not self.plate.text():
            do = False
            QtWidgets.QMessageBox.warning(self,
                                    "Plate not set!",
                                    "Type a plate to search for.",
                                    QtWidgets.QMessageBox.Ok
                                    )
                                    
        if do:
            plateText = self.plate.text().replace(" ", "").upper()
            if self.curId is not None: # Update a previous row
                DB.execute("UPDATE targetPlates SET name=?, plate=? WHERE id=?", (self.name.text(), plateText, self.curId) )
                self.parent.loadPlateTargets() # refresh list
                self.parent.parent.reInitializeProcesses()
                self.close()
            else:
                try:
                    DB.execute("INSERT INTO targetPlates (name, plate) VALUES (?,?)", (self.name.text(), plateText) )
                except sql.IntegrityError:
                    QtWidgets.QMessageBox.warning(self,
                                    "Plate already present!",
                                    "This plate is already in database.",
                                    QtWidgets.QMessageBox.Ok
                                    )
                    return
                self.parent.loadPlateTargets() # refresh list
                self.parent.parent.reInitializeProcesses()
                self.close()
                    
class addFace(QtWidgets.QDialog):
    def __init__(self, parent, faceId=None):
        super().__init__(parent) #inheriting from the object.
        uic.loadUi(os.path.join(settings.CUR_PATH,'GUI','aggiungifaccia.ui'), self)
        self.openFileButton.clicked.connect(self.fileDialog)
        self.parent = parent
        self.saveButton.clicked.connect(self.save)
        self.filenames = None
        self.curId = None
        self.keepPreviousTemplate = False
        # Load current face info
        if faceId is not None:
            res = DB.execute("SELECT name, faces FROM targetFaces WHERE id=(?) LIMIT 1", (faceId,)).fetchone()
            self.curId = faceId
            self.name.setText(res[0])
            self.keepPreviousTemplate = True
            self.labelFileTip.setText("One or more faces are already set for this target. Inserting new images will delete all the previous images.")
        
    def fileDialog(self):
        self.filenames = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select one or more faces', settings.CUR_PATH, "Images (*.png *.jpg *.jpeg)")        
        if len(self.filenames[0])>0:
            self.labelFileTip.setText(str(len(self.filenames[0]))+" photo(s) selected.")
            self.keepPreviousTemplate = False
            
    def save(self):
        do = True
        if not self.filenames[0] and not self.keepPreviousTemplate:
            do = False
            QtWidgets.QMessageBox.warning(self,
                                    "Face(s) not set!",
                                    "Select at least a image file containing a face to analyse.",
                                    QtWidgets.QMessageBox.Ok
                                    )
        if not self.name.text():
            do = False
            QtWidgets.QMessageBox.warning(self,
                                    "Name not set!",
                                    "Type a name as reference for this face.",
                                    QtWidgets.QMessageBox.Ok
                                    )
        if do:
            template = [] # List of templates
            if self.filenames[0] and not self.keepPreviousTemplate:
                for path in self.filenames[0]:
                    if not os.path.isfile(path):
                        QtWidgets.QMessageBox.warning(self, "File not found!", "One of selected files was not found!", QtWidgets.QMessageBox.Ok)
                        break
                    img = cv2.imread(path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    rect = FACE_DETECTOR(img, 1) # Faces found
                    if len(rect)!=1:
                        QtWidgets.QMessageBox.warning(self, "Photo not good!", "There are no faces or there are 2 or more faces in the same photo. Cut every image to contain only one face and repeat.", QtWidgets.QMessageBox.Ok)
                        self.filenames[0] = None
                        break
                    else:
                        landmarks = FACE_POSE_PREDICTOR(img, rect[0]) # 68 landmarks
                        measures = np.array(FACE_RECOGNITION_MODEL.compute_face_descriptor(img, landmarks)) # 128 measures
                        template.append(measures.tolist())
            
            if self.curId is not None: # Update a previous row
                if self.keepPreviousTemplate: # Update only name
                    DB.execute("UPDATE targetFaces SET name=? WHERE id=?", (self.name.text(), self.curId) )
                    self.parent.loadFaceTargets() # refresh list
                    self.parent.parent.reInitializeProcesses()
                    self.close()
                elif template: # Update name and template
                    DB.execute("UPDATE targetFaces SET name=?, faces=? WHERE id=?", (self.name.text(), json.dumps(template), self.curId) )
                    self.parent.loadFaceTargets()
                    self.parent.parent.reInitializeProcesses()
                    self.close()
            elif template:  # Create a new row     
                DB.execute("INSERT INTO targetFaces (name, faces) VALUES (?,?)", (self.name.text(), json.dumps(template)) )
                self.parent.loadFaceTargets()
                self.parent.parent.reInitializeProcesses()
                self.close()
            
class about(QtWidgets.QWidget):
    def __init__(self,parent):
        super().__init__()
        self.parent = parent
        uic.loadUi(os.path.join(settings.CUR_PATH,'GUI','about.ui'), self)
        self.buttonOk.clicked.connect(self.close)
        centerOnScreen(self) # Centers window on the screen

class analyzeFile(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        uic.loadUi(os.path.join(settings.CUR_PATH,'GUI','analizzafile.ui'), self)
        self.doButton.clicked.connect(self.start)
        self.openFileButton.clicked.connect(self.fileDialog)
        self.saveReport.clicked.connect(self.outputDialog)
        self.buttonCancel.clicked.connect(self.close)
        self.selectROI.clicked.connect(self.openRoiSelection)
        self.unsetROI.clicked.connect(self.doUnsetRoi)
        self.thread = analyzeThread(parent=self)
        self.thread.progress_update.connect(self.updateProgressBar)
        self.thread.finish.connect(self.finishMessage)
        self.filenames = [None]
        self.output = [None]
        centerOnScreen(self) # Centers window on the screen
        
        
    def fileDialog(self):
        self.filenames = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select one or more video files', settings.CUR_PATH, "Videos (*.avi *.mp4 *.mov *.mpeg *.mpg *.flv *.wmv)")        
        if len(self.filenames[0])>0:
            self.labelFileTip.setText(str(len(self.filenames[0]))+" file(s) selected.")
    
    def outputDialog(self):
        self.output = QtWidgets.QFileDialog.getSaveFileName(self, 'Select output path', os.path.join(settings.CUR_PATH,'ReportFaceGMT.csv'), "Text (*.csv)")        
        
    def closeEvent(self, event):
        if self.thread.isRunning():
            reply = QtWidgets.QMessageBox.question(self, 'Confirmation', "Do you REALLY want to stop the file analysis process?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.thread.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
    
    def start(self):
        if len(self.filenames[0])>0 and self.output[0]:
            self.thread.set(self.filenames[0], self.output[0], self.saveNewFace.isChecked(), self.saveNewPlate.isChecked())
            self.thread.start()
            self.enable(False)
        else:
            QtWidgets.QMessageBox.warning(self,
                                    "Missing values!",
                                    "There are some missing values. Please check to have every field set.",
                                    QtWidgets.QMessageBox.Ok
                                    )
        
    def enable(self, status=True):
        self.saveNewFace.setEnabled(status)
        self.saveNewPlate.setEnabled(status)
        self.doButton.setEnabled(status)
        self.openFileButton.setEnabled(status)
        self.saveReport.setEnabled(status)
    
    def updateProgressBar(self, value):
        self.progressBar.setValue(value)
    
    def finishMessage(self, value):
        self.enable(True)
        if value:
            QtWidgets.QMessageBox.information(self,
                                        "Analysis completed!",
                                        "The file analysis has terminated. You'll find the report in the selected path.",
                                        QtWidgets.QMessageBox.Ok
                                        )
        else:
            QtWidgets.QMessageBox.critical(self,
                                        "Analysis failed!",
                                        "The file analysis has terminated with ERRORS. Please see the report for details.",
                                        QtWidgets.QMessageBox.Ok
                                        )
    
    def openRoiSelection(self):
        try:
            cap = Capture(source=self.filenames[0])
            cap.start()
            frame=None
            i=0
            while(frame is None and i<5):
                time.sleep(1)
                frame = cap.get()
                i+=1
            cap.stop()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self,
                                    "Cannot open file!",
                                    "Please select a file to process.",
                                    QtWidgets.QMessageBox.Ok
                                    )
            return
        else:
            if frame is None:
                QtWidgets.QMessageBox.warning(self,
                                        "No video!",
                                        "Please select a valid file.",
                                        QtWidgets.QMessageBox.Ok
                                        )
                return
            
            cv2.namedWindow('Set ROI', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Set ROI',0,0)
            cv2.resizeWindow('Set ROI', 800,600)
            r = cv2.selectROI('Set ROI',frame,True,False)
            cv2.destroyWindow('Set ROI')
            if r and r[2] and r[3]:
                self.roiValue.setText( '{} {} {} {}'.format(r[0], r[1], r[2], r[3]) )
            
    def doUnsetRoi(self):
        self.roiValue.setText('')
        
        
                                    
class analyzeThread(QtCore.QThread):

    progress_update = QtCore.pyqtSignal(int)
    finish = QtCore.pyqtSignal(bool)

    def __init__(self, parent):
        QtCore.QThread.__init__(self)
        self.files = []
        self.doNewFaces = False
        self.doNewPlates = False
        self.output = None
        self.parent = parent
        
    def __del__(self):
        self.wait()
    
    def set(self, files, output, doNewFaces, doNewPlates):
        self.files = files
        self.output = output
        self.doNewFaces = doNewFaces
        self.doNewPlates = doNewPlates
    
    def run(self):
        roiValueText = self.parent.roiValue.text()
        if roiValueText: # Consider ROI
            r = [int(s) for s in roiValueText.split() if s.isdigit()]
            roiValue = [r[0], r[1], r[0]+r[2], r[1]+r[3]] # x1, y1, x2, y2
        else:
            roiValue = [None,None,None,None]
        
        DB = sql.connect(settings.DB_PATH, isolation_level=None) # Open connection (automatically creates file if does not exist) in AUTOCOMMIT MODE
        # Load Target Faces
        targetFaces_data = DB.execute("SELECT id, faces, name FROM targetFaces").fetchall() # Load targetFaces data
        targetFaces = [ [row[0], [np.array(t) for t in json.loads(row[1])], row[2] ] for row in targetFaces_data ] # Build a list and convert templates to JSON
        
        # Load Target Plates
        targetPlates = DB.execute("SELECT id, name, plate FROM targetPlates").fetchall() # Load targetPlates data
        
        imageOutputDir = os.path.join(os.path.dirname(self.output),os.path.splitext(os.path.basename(self.output))[0]+'_images')
        os.makedirs(imageOutputDir, exist_ok=True)
        csvfile = open(self.output, 'w')
        writer = csv.writer(csvfile, delimiter=';', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(['FILE', 'TIME', 'TYPE', 'TARGET', 'PLATE', 'FRAME'])
        csvfile.flush()
        
        doneDuration = 0
        totalDuration = 0
        # Get stats
        for f in self.files:
            try:
                cap = cv2.VideoCapture(f)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                totalDuration += frame_count/fps
                cap.release()
            except:
                pass
        
        numCpu = mp.cpu_count()
        pool = mp.Pool(processes=max(numCpu,1))
        manager = mp.Manager()
        frameQueue = manager.JoinableQueue(numCpu) # Queue with max number of frames (max size is numCpu!)
        resQueue = manager.Queue() # Queue with returning rows
        # Start sub processes
        for i in range(numCpu):
            pool.apply_async(processingFrame, args=(frameQueue, resQueue, targetFaces, targetPlates, self.doNewFaces, self.doNewPlates, imageOutputDir, roiValue), error_callback=self.workerError)
            
        # Do processing
        for f in self.files:
            filename = os.path.basename(f)
            try:
                cap = cv2.VideoCapture(f)
                fps = cap.get(cv2.CAP_PROP_FPS)
                count = 0
                doProcess = True
                while doProcess:
                    
                    ret, frame = cap.read()
                    if ret:
                        frameQueue.put([frame, count, fps, filename]) # Waits if frameQueue is full
                        count+=1
                        self.progress_update.emit( int( 100*(doneDuration + count/fps) / totalDuration ) )
                    else:
                        frameQueue.join() # Waits JoinableQueue.task_done() on all elements.
                        doProcess = False
                        
                    while True: # Write previous results
                        try:
                            res = resQueue.get(block=False)
                            for row in res: # processingFrame returns a list of rows
                                writer.writerow(row)    
                        except queue.Empty:
                            break
                        except Exception as e:
                            writer.writerow([filename,'','-------- ERROR --------' + str(e)])    
                    
                # Prepare for file change
                #pool.close()            
                doneDuration += count/fps
                # Close streaming
                cap.release()
            
            except Exception as e:
                # Write error in report! HERE
                writer.writerow([filename,'','-------- ERROR --------' + str(e)])
                continue
            
            
        if doneDuration/totalDuration==1:
            msg = '-------- ANALYSIS COMPLETED WITH SUCCESS -------- ' + str(doneDuration)+'s / '+str(totalDuration)+'s'
            if roiValueText:
                msg +=' WITH ROI: '+roiValueText
            writer.writerow([msg])
            self.finish.emit(True)
        else:
            msg = '-------- ANALYSIS COMPLETED WITH MISSING PARTS -------- ' + str(doneDuration)+'s / '+str(totalDuration)+'s'
            if roiValueText:
                msg +=' WITH ROI: '+roiValueText
            writer.writerow([msg])
            self.finish.emit(False)
        
        csvfile.flush()
    
    
    def workerError(self, e): # CRITICAL ERROR (not managed) IN WORKER. IMMEDIATE STOP.
        QtWidgets.QMessageBox(parent = self.parent, icon = QtWidgets.QMessageBox.Critical,
                    windowTitle="Unexpected error!",
                    text="A unexpected error has occurred in a video sub-process! Results may be corrupted.",
                    standardButtons=QtWidgets.QMessageBox.Ok,
                    detailedText=repr(e)
                    ).exec_()
        

################################## INNER FUNCTION START #################################
def processingFrame(frameQueue, resQueue, targetFaces, targetPlates, doNewFaces, doNewPlates, imageOutputDir, roiValue):
    while True:
        frameData = frameQueue.get() # Waits for frameData
        frame = frameData[0]
        count = frameData[1]
        fps = frameData[2]
        filename = frameData[3]
        output = []
        # PROCESSING
        if frame is not None:
            saveFrame = False
            frameName = filename+'_'+str(count)+'.png'
            frame = frame[roiValue[1]:roiValue[3],roiValue[0]:roiValue[2]] # Cut to ROI (if x1,y1,x2,y2 are None, frame remains the same)
            # FACE RECOGNITION
            detected_faces = FACE_DETECTOR(frame, 1) # Detect faces (quite slow)
            if len(detected_faces)>0:
                for i, rect in enumerate(detected_faces): # For every detected face
                    ########### SLOW PART > 0.3 s per face ##########
                    landmarks = FACE_POSE_PREDICTOR(frame, rect) # Get 68 points
                    measures = FACE_RECOGNITION_MODEL.compute_face_descriptor(frame, landmarks) # Get 128 measures
                    ########### SLOW PART END ##############
                    dist = 1 # Distance
                    bestMatch = None # Best match found
                    for t in targetFaces: # For every target in database
                        for f in t[1]: # In every template
                            if any(f): # Avoid empty templates
                                new_dist = np.linalg.norm(measures - f)
                                if new_dist < settings.MAX_DISTANCE and new_dist < dist:
                                    dist = new_dist # Update best match
                                    bestMatch = t
                    if bestMatch is not None:
                        saveFrame = True
                        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.left()+rect.width(), rect.top()+rect.height()), (0, 0, 255), 2) # Draw RED rectangle around the faces
                        cv2.putText(frame, bestMatch[2],(rect.left(),rect.top()), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                        cv2.putText(frame,"{:.0%}".format(1-dist),(rect.left(),rect.bottom()), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
                        output.append([filename, humanize_time(count/fps), 'F', t[2], '', frameName])
                    elif doNewFaces:
                        saveFrame = True
                        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.left()+rect.width(), rect.top()+rect.height()), (0, 255, 0), 2) # Draw GREEN rectangle around the faces
                        output.append([filename, humanize_time(count/fps), 'F', '', '', frameName])
                        
            # PLATE RECOGNITION
            bestPlate = searchBestPlate(frame, ALPR)
            if bestPlate:
                bestPlate = bestPlate.upper()
                targetData = None
                for tar in targetPlates: # Search in targets
                    if bestPlate == tar[2].upper():
                        targetData = tar
                        break
                # Save to db
                if targetData:
                    output.append([filename, humanize_time(count/fps), 'P', targetData[1], bestPlate, frameName])
                    saveFrame = True
                elif doNewPlates:
                    output.append([filename, humanize_time(count/fps), 'P', '', bestPlate, frameName])
                    saveFrame = True
                    
            if saveFrame:    
                # Save image in folder too!        
                cv2.imwrite( os.path.join(imageOutputDir, frameName), frame )
                resQueue.put(output)
        
        frameQueue.task_done()
    
################################## INNER FUNCTION END #################################

            
# Main separate recognition process
def recognitionProcess(camId):
    DB = sql.connect(settings.DB_PATH, isolation_level=None) # Open connection (automatically creates file if does not exist) in AUTOCOMMIT MODE
    cam = DB.execute("SELECT url, saveNewFaces, saveNewPlates, activeFace, activePlate, roi FROM cameras WHERE id = ?", (camId,) ).fetchone()
    saveNewFaces = cam[1]
    saveNewPlates = cam[2]
    doFace = cam[3]
    doPlate = cam[4]
    if cam[5]: # Show ROI
        r = [int(s) for s in cam[5].split() if s.isdigit()]
        x1 = r[0]
        y1 = r[1]
        x2 = r[0]+r[2]
        y2 = r[1]+r[3]
    else:
        x1 = None
        y1 = None
        x2 = None
        y2 = None
        
    cap = Capture(source=cam[0])
    cap.start()
    
    savePath = os.path.join(settings.EVENTS_PATH, str(camId))
    os.makedirs(savePath, exist_ok=True)
    
    # Load Target Faces
    targetFaces_data = DB.execute("SELECT id, faces, name FROM targetFaces").fetchall() # Load targetFaces data
    targetFaces = [ [row[0], [np.array(t) for t in json.loads(row[1])], row[2] ] for row in targetFaces_data ] # Build a list and convert templates to JSON
    
    # Load Target Plates
    targetPlates = DB.execute("SELECT id, name, plate FROM targetPlates").fetchall() # Load targetPlates data
    
    
    while(True):
        frame = cap.get()
        frameTime = datetime.datetime.now()
        if frame is not None:
            frame = frame[y1:y2,x1:x2] # Cut to ROI (if x1,y1,x2,y2 are None, frame remains the same)
            saveFrame = False
            # FACE RECOGNITION
            if doFace:
                detected_faces = FACE_DETECTOR(frame, 1) # Detect faces (quite slow)
                if len(detected_faces)>0:
                    for i, rect in enumerate(detected_faces): # For every detected face
                        ########### SLOW PART > 0.3 s per face ##########
                        landmarks = FACE_POSE_PREDICTOR(frame, rect) # Get 68 points
                        measures = FACE_RECOGNITION_MODEL.compute_face_descriptor(frame, landmarks) # Get 128 measures
                        ########### SLOW PART END ##############
                        dist = 1 # Distance
                        bestMatch = None # Best match found
                        for t in targetFaces: # For every target in database
                            for f in t[1]: # In every template
                                if any(f): # Avoid empty templates
                                    new_dist = np.linalg.norm(measures - f)
                                    if new_dist < settings.MAX_DISTANCE and new_dist < dist:
                                        dist = new_dist # Update best match
                                        bestMatch = t
                        if bestMatch is not None:
                            saveFrame = True
                            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.left()+rect.width(), rect.top()+rect.height()), (0, 0, 255), 2) # Draw RED rectangle around the faces
                            cv2.putText(frame, bestMatch[2],(rect.left(),rect.top()), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                            cv2.putText(frame,"{:.0%}".format(1-dist),(rect.left(),rect.bottom()), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
                            DB.execute("INSERT INTO eventFaces (camera, datetime, target) VALUES (?,?,?)", (camId, frameTime.strftime('%Y%m%d%H%M%S%f'), bestMatch[0]) )
                        elif saveNewFaces:
                            saveFrame = True
                            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.left()+rect.width(), rect.top()+rect.height()), (0, 255, 0), 2) # Draw GREEN rectangle around the faces
                            DB.execute("INSERT INTO eventFaces (camera, datetime) VALUES (?,?)", (camId, frameTime.strftime('%Y%m%d%H%M%S%f')) )
                
            # PLATE RECOGNITION
            if doPlate:
                bestPlate = searchBestPlate(frame, ALPR)
                if bestPlate:
                    bestPlate = bestPlate.upper()
                    idTarget = None
                    for tar in targetPlates: # Search in targets
                        if bestPlate == tar[2].upper():
                            idTarget = tar[0]
                            break
                    # Save to db
                    if idTarget:
                        DB.execute("INSERT INTO eventPlates (camera, datetime, plate, target) VALUES (?,?,?,?)", (camId, frameTime.strftime('%Y%m%d%H%M%S%f'), bestPlate, idTarget) )
                        saveFrame = True
                    elif saveNewPlates:
                        DB.execute("INSERT INTO eventPlates (camera, datetime, plate) VALUES (?,?,?)", (camId, frameTime.strftime('%Y%m%d%H%M%S%f'), bestPlate) )
                        saveFrame = True
                    
            if saveFrame:    
                # Save image in folder too!        
                cv2.imwrite( os.path.join(savePath,frameTime.strftime('%Y%m%d%H%M%S%f.png')), frame )
            
    
        
# Common Utilities
def centerOnScreen(w): # Centers the window on the screen
    resolution = QtWidgets.QDesktopWidget().screenGeometry()
    w.move((resolution.width() / 2) - (w.frameSize().width() / 2),
              (resolution.height() / 2) - (w.frameSize().height() / 2))
    w.setFixedSize(w.size()) # Fixed dimensions (how to be responsive?)

def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d' % (hours, mins, secs)

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def searchBestPlate(frame, alprObj):
    plateRes = alprObj.recognize_ndarray(frame)
    if plateRes and plateRes["results"] and plateRes["results"][0]["confidence"] > settings.OPENALPR_MIN_CONFIDENCE:
        return plateRes["results"][0]["plate"]
    else:
        for angle in settings.OPENALPR_ROTATIONS:
            frame_rot = rotate_image(frame, angle)
            plateRes = alprObj.recognize_ndarray(frame_rot)
            if plateRes and plateRes["results"] and plateRes["results"][0]["confidence"] > settings.OPENALPR_MIN_CONFIDENCE:
                return plateRes["results"][0]["plate"]
        return None
            
def initialChecks(parent = None):
    forceExit = False
    # Check db
    if not os.path.isfile(settings.DB_PATH):
        msg = QtWidgets.QMessageBox(parent = parent, icon = QtWidgets.QMessageBox.Critical, windowTitle="Error!", text="Database not found!", standardButtons=QtWidgets.QMessageBox.Ok)
        msg.setDetailedText("Database must be in %s" % settings.DB_PATH)
        msg.exec_()
        forceExit = True
    
    # Exit flag
    if forceExit:
        sys.exit(0)
    

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv) # Start GUI
    initialChecks() # Do initial checks (after app instance)
    DB = sql.connect(settings.DB_PATH, isolation_level=None) # Open connection (automatically creates file if does not exist) in AUTOCOMMIT MODE
    window = mainWindow() # Keep reference to main window
    window.show() # Open main window
    window.recognitionInitialization() # START ALL THE BACKGROUND PROCESSES
    sys.exit(app.exec_())
    
