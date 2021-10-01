from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)


import time
import numpy as np
import cv2 #OpenCV module for detection
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import pandas as pd

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5 # If cosine distance between two points is less than 0.5, model will consider them the same object
nn_budget = None
nms_max_overlap = 0.8 # avoids multiple dectections of the same object

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('.\\Enter_Directory_here.mp4')
#vid = cv2.VideoCapture(0)

codec = cv2.VideoWriter_fourcc(*'XVID')# output file format will be avi. MJPEG and other codecs are available
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))*10 #Use the openCV functionality to extract the fps of video and convert to int
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) # get video width and height. Put to int
out = cv2.VideoWriter('./data/video/dump.avi', codec, vid_fps, (vid_width, vid_height))

# Variables later used to record the number of people by track_id's
counter = []
current_count = int(0)
daily_count = []
daily_average = 0
people_list = []
av_people = 0

location = 'NaN' #To be captured form API
weather = 'NaN'	#To be captured form API
temprature='NaN'	#To be captured form API
count_dict = [0]
daily_average_dict = {}
av_people_list = []

while True:
    #Capture the images in the video
	_, img = vid.read()
    #If there exists no more frames, mark as completed
	if img is None:
		print("Completed")
		break
	#Transform images as suitable for opencv and tensorflow models
	img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_in = tf.expand_dims(img, 0)
	img_in = transform_images(img_in, 416)
 
	#Intial time
	t1 = time.time()
	
	boxes, scores, classes, nums = yolo.predict(img_in)
	classes = classes[0]
	names = []
	for i in range(len(classes)):
		names.append((class_names[int(classes[i])]))
  
	names = np.array(names)
	converted_boxes = convert_boxes(img, boxes[0])
	features = encoder(img, converted_boxes)
	detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
	boxs = np.array([d.tlwh for d in detections])
	scores = np.array ([d.confidence for d in detections])
	classes = np.array([d.class_name for d in detections])
	indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores) # measure of which boxes should be removed during multiple frames in an image
	detections = [detections[i] for i in indices] #use indices value to remove redundancy from the detections
	tracker.predict()
	tracker.update(detections)
 
	#Color Organizing
	cmap = plt.get_cmap('tab20b') # get a range of colors
	colors = [cmap(i)[:3] for i in np.linspace(0,1,20)] #set a range of colors with increment
	
 
	for track in tracker.tracks:
		if not track.is_confirmed() or track.time_since_update >1: #if the model didn't detect anything or if time runs out, pass the frame
			continue
		if not str(track.get_class()) == 'person': #only selecting the people class
			continue

		bbox = track.to_tlbr() #configure the boxes' corrdinates
		class_name= track.get_class() #fetch classes from the frames
		color = colors[int(track.track_id) % len(colors)] #set colors
		color = [i * 255 for i in color] #convert to RGB since they were initially from 0,1
		cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)# set the boxes' corrdinates's extrema
		cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1) #set boundary the class name and changing the colors to be displayed on thebox
		cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,(255, 255, 255), 2)# write the names of the track_id, and class name appropriatel
		count_dict.append(int(track.track_id))
	
		if len(count_dict) % (6) == 0:#Check the number of people every 5 mins
			for i in range(len(count_dict)):
				total_people = 0
				total_people += count_dict[i] #adds all the number of people in the multiple of four frames
				people_list.append(total_people)
				av_people_list.append(int(people_list[i]/len(people_list)))
				daily_count.append(total_people)
				daily_average_dict[str(datetime.now().time())] = av_people_list[i]
				
			print(people_list)
			av_people = np.ceil(np.mean(people_list))
			print(av_people)	
		if str(datetime.now().strftime('%X')) == '23:59:59':
			daily_max = max(av_people_list)
			daily_min = min(av_people_list)
			daily_min_time = list(av_people_list.keys())[list(av_people_list.values()).index(daily_min)]
			daily_max_time = list(av_people_list.keys())[list(av_people_list.values()).index(daily_man)]
			cv2.putText(img, "DAILY AVERAGE: {} people". format(daily_average), (400,700), 0,1, (211,211,211), 1)
			#Write daily Statistics to a CSV file when the clock is just about to hit midnight
			with open('.\\daily_statistics.csv', 'a', newline='\n') as daily:
				writer = csv.writer(file, delimiter = ',')
				header_fields = ["Day Of Week", "Location", "Weather", "Daily_Average", "Daily_Max", 'Daily_Max_time', "Daily_min", "Daily_min_time"]
				input_fields = [str(datetime.now()), str(datetime.now().strftime("%a")),Weather,daily_average,daily_max, daily_max_time, daily_min,daily_min_time]
				try:
					pd.read_csv('.\\daily_statistics.csv')
					writer.writerow(input_fields)
				except Exception as e:
					if str(e) == 'pandas.errors.EmptyDataError' or str(e) == 'No columns to parse from file':
						writer.writerow(header_fields)
					else:
						print(str(e))
		
  
		#Write 5 minute data to CSV
		with open('.\\per_5_min_data.csv', 'a', newline = '\n') as file:
			writer = csv.writer(file, delimiter = ',')
			header_fields = ['Date', 'Day of Week', 'Location', 'Weather', 'Average People per {} mins'. format(int(1*vid_fps/vid_fps))]
			input_fields = [str(datetime.now()), str(datetime.now().strftime("%a")), location, weather, av_people]
			try:
				pd.read_csv('.\\per_5_min_data.csv')
				writer.writerow(input_fields)
			except Exception as e:
				if str(e) == 'pandas.errors.EmptyDataError' or str(e) == 'No columns to parse from file':
					writer.writerow(header_fields)
				else:
					print(str(e))
			
			
		
		if not count_dict: # if the list is empty i.e. the first entry
			count_dict[0] == 0

		cv2.putText(img, "Current: " + str(count_dict[-1]) , (400, 60), 0, 1, (0, 0, 255), 1)
		cv2.putText(img, "Average: {} ".format(av_people), (400, 30), 0, 1, (0, 0, 255), 1)
		output = 'output'
		fps = 1./(time.time()-t1)
		cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
		cv2.imshow('output', img)
		cv2.resizeWindow(str(output),1024, 768)
		out.write(img)
	if cv2.waitKey(1) == 27:
		break
     
vid.release()
out.release()
cv2.destroyAllWindows()