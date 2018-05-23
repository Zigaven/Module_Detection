#!/usr/bin/python
# -*- coding: utf-8 -*-


# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import threading
from re import findall

import requests as requests
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import timeit
import cv2
import requests
import urllib.request
import json
import datetime
import uuid
import subprocess as sp


class Incident(object):
    # A class attribute. It is shared by all instances of this class
    species = "H. sapiens"

    # Basic initializer, this is called when this class is instantiated.
    # Note that the double leading and trailing underscores denote objects
    # or attributes that are used by python but that live in user-controlled
    # namespaces. You should not invent such names on your own.
    def __init__(self, name, start_time, finish_time, device, periodicity, chat_id, camera, notification):
        self.name = name
        self.start_time = start_time
        self.finish_time = finish_time
        self.device = device
        self.periodicity = periodicity
        self.chat_id = chat_id
        self.camera = camera
        self.notification = notification

    @staticmethod
    def create_from_dict(data):
        return Incident(data["name"], data["start_time"], data["finish_time"], data["device"],
                      data["periodicity"], data["chat_id"], data["camera"], data["notification"])



def send_message(chat_id, message):
    url = "https://api.telegram.org/bot586715397:AAGOCEMPP9s5qNGLT1G1aNdrJ1NOFeA5Yj0/sendMessage?chat_id=" + chat_id + "&text=" + message

    contents = urllib.request.urlopen(url).read()
    print(contents)


def get_incidents():
    with urllib.request.urlopen("http://localhost:8888/get.php") as url:
        data = json.loads(url.read().decode())
        incidents = []
        for id, incident in data.items():
            incidents.append(Incident.create_from_dict(incident))

        print(len(incidents))
        return incidents

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
k = 0;
g = True
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")


vs = VideoStream(src=0).start()
# vs2 = VideoStream(src="rtmp://10.17.3.128:1935/live/dimon").start()
# videos = [vs,vs2]
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
# for video in videos:


command = ['ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec','rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', '1124x800',
    # '-use_wallclock_as_timestamps', '1',
    '-i', '-',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-framerate', '30',
    '-f', 'flv',
    'rtmp://10.17.46.217:1935/live/huila']






while True:

    incidents = get_incidents()

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels

    frame = vs.read()
    frame = imutils.resize(frame, width=1124, height=800)
      # frame is read using opencv


    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)

            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output frame
    word = findall(r'\w+', label)[0]
    #     print(word)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop

    current_time = datetime.datetime.now()


    for inc in incidents:
        start = datetime.datetime.strptime(inc.start_time, "%Y-%m-%d %H:%M:%S")
        end = datetime.datetime.strptime(inc.finish_time, "%Y-%m-%d %H:%M:%S")

        if start <= current_time <= end:
            if word == "person" and g == True:
                # cv2.imshow("Frame", frame)
                k = k + 1
                if inc.device == "phone":
                    t1 = threading.Thread(target=send_message, args=(inc.chat_id, "Outside activity detected at " + str(current_time)))
                    t1.start()
                filename = str(uuid.uuid4()) + ".jpg"
                cv2.imwrite(filename, frame)
                file = {'media': open(filename, 'rb')}
                print(inc.device)
                if inc.device == "phone":
                    r = requests.post("http://localhost:8888/info.php", data={'detection_time':str(current_time), 'chat_id': inc.chat_id, 'message': 'Outside activity detected at'}, files=file)

                # t1.join()
                print("Обнаружена постороняя активность")
                print("Обнаружение объекта: ", current_time)
                # print(incidents.chat_id)
                g = False
            # telegram part

            if word != "person" and g == False:
                # print("FOUND PERSON")
                print("Объект исчез: ", current_time)
                if inc.device == "phone":
                    t2 = threading.Thread(target=send_message, args=(inc.chat_id, "Outside activity has disappeared at " + str(current_time)))
                    t2.start()
                if inc.device == "Web-Application":
                    r2 = requests.post("http://localhost:8888/info.php",
                                  data={'detection_time': str(current_time), 'chat_id': inc.chat_id,
                                        'message': 'Outside activity has disappeared at'})
                # t2.join()

                g = True
                # break

            # telegram part2

    if key == ord("q"):
        break
# update the FPS counter
fps.update()

proc = sp.Popen(command, stdin=sp.PIPE,shell=False)
proc.stdin.write(frame)


# while True:
#     try:
#         check_updates()
#     except KeyboardInterrupt:  # порождается, если бота остановил пользователь
#         print('Interrupted by the user')
#         break
# if cv2.imshow("Frame", frame):
# print("Found")
# stop the timer and display FPS information
print("Количество появление объектов:", k)
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
