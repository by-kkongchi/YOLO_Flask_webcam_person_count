import numpy as np
import cv2

def detectObjects(video_path):

    confidenceThreshold = 0.5
    NMSThreshold = 0.3

    modelConfiguration = 'cfg/yolov3.cfg'
    modelWeights = 'yolov3.weights'

    labelsPath = 'coco.names'
    labels = open(labelsPath).read().strip().split('\n')

    np.random.seed(10)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    i = net.getUnconnectedOutLayers()
    outputLayer = net.getLayerNames()
    outputLayer = [outputLayer[i[0] - 1]]

    video_capture = cv2.VideoCapture(0)

    (W, H) = (None, None)

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        if W is None or H is None:
            (H,W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB = True, crop = False)
        net.setInput(blob)
        layersOutputs = net.forward(outputLayer)

        boxes = []
        confidences = []
        classIDs = []

        for output in layersOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidenceThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY,  width, height) = box.astype('int')
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        #Apply Non Maxima Suppression
        detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)  
        outputs ={}
        if len(detectionNMS) > 0 :
            outputs['detections'] ={}
            outputs['detections']['labels'] =[]
            for i in detectionNMS.flatten():
                detection ={}
                detection['classIDs[i]']= labels[classIDs[i]]
                detection['confidence']= confidences[i]
                #detection['X'] = boxes[i][0]
                #detection['Y'] = boxes[i][1]
                #detection['Width'] = boxes[i][2]
                #detection['Height'] =boxes[i][3]
                outputs['detections']['labels'].append(detection)
        else:
            outputs['detections'] = 'No object detected'

        
        
        return outputs
        
      
