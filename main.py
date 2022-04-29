import cv2
thres = 0.5 # Threshold to detect object
 
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)
 
classNames= []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")
 
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"
 
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 255)
net.setInputMean((255, 255, 255))
net.setInputSwapRB(True)
 
while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(bbox)
    w, h, c = img.shape
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(), bbox):
            cv2.rectangle(img, box,color=(0,255,0), thickness=2)
            cv2.putText(img, classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img, "x,y-({},{})".format(box[0],box[1]), (box[0]+10, box[3]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)
            cv2.putText(img, "w,h-({},{})".format(w,h), (box[0]+250, box[3]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2, )

            print("{} x-{}".format(classId, box[0]))
            print("{} y-{}".format(classId, box[1]))
            print("width-{}".format(w))
            print("hight-{}".format(h))
            print("---------------------------------------------------")

    cv2.imshow("Output", img)
    cv2.waitKey(1)