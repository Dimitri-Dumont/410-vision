import cv2 as cv
import numpy as np
import glob


def detectObject(img):
 
    classNames= []
    classFile = 'coco.names.txt'
    with open(classFile,'rt') as f:
        classNames = f.read().split('\n')
    
    #print(classNames)
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, conf, bbox = net.detect(img, confThreshold=.6)
    print(classIds,bbox)

    if len(classIds != 0):
        for classId, confidence, box in zip(classIds.flatten(), conf.flatten(),bbox):
            cv.rectangle(img,box,color=(255,255,0),thickness=2)
            cv.putText(img,classNames[classId - 1]
            ,(box[0]+10,box[1]+30),cv.FONT_HERSHEY_COMPLEX ,1,(255,255,0),2)

    cv.imshow('test',img )
    cv.waitKey(0)

if __name__ == "__main__":
   
    path = glob.glob("*.jpg")
    cv_img = []
    for img in path:
        n = cv.imread(img)
        cv_img.append(n)
    for img in cv_img:
        detectObject(img)