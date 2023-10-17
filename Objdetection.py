# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import cv2
# import os

# os.system("sudo modprobe bcm2835-v4l2")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

'''CLASSES = ["", "", "", "", "",
    "bottle", "", "", "", "", "", "",
    "", "", "", "person", "", "",
    "", "", ""]'''

COLORS = np.random.uniform(0, 255, size=(len(CLASSES)))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
fps = FPS().start()
ph= 0
pw = 0
# loop over the frames from the video stream
a,b,c,d,e,f = 1,1,1,1,1,1

while (True):
    frame = vs.read()
    cv2.imwrite('test.jpg',frame)
    frame = cv2.imread('test.jpg')
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if(confidence > 0.2):
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = CLASSES[idx]
            cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            if label == 'car':
                if (endY < 240):
                    if (endX < 213):a = 0
                    elif (endX < 426):b = 0
                    elif(endX < 640):c = 0

                else:
                    if (endX < 213):d = 0
                    elif (endX < 426):e = 0
                    else:f = 0

        print (a,b,c,d,e,f)

    # show the output frame
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
    fps.update()
    count = 0
    a, b, c, d, e, f = 1, 1, 1, 1, 1, 1

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

    
