import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
from sort import *
tracker = Sort()
from deep_sort.liu_detection import Detection
import math
'''
#179 10
line1 = [(32,113), (136,141)]
line2 = [(1,137), (89,168)]
line3 = [(149,145), (283,172)]
line4 = [(180,124), (285,142)]
#Daya 10
line1 = [(82,286), (326,315)]
line2 = [(9,305), (264,341)]
line3 = [(307,332), (673,393)]
line4 = [(387,315), (674,358)]
'''

#benchmark
#line1 = [(504,554), (1009,553)]
line1 = [(499,576), (1028,581)]
line2 = [(430,662), (1077,664)]
line3 = [(1143,613), (1808,639)]
#line4 = [(1062,526), (1785,567)]
line4 = [(1087,550), (1810,581)]

'''#old 1080
line1 = [(527,625), (1050, 621)]
line2 = [(416,791), (1130, 786)]
line3 = [(1250,724), (1950, 731)]
line4 = [(1140,598), (1750, 609)]
'''
#taichung1
#line1 = [(95,166), (172, 161)]
#line2 = [(102,195), (206, 184)]
#line3 = [(261,175), (301, 155)]
#line4 = [(226,161), (269, 141)]
#taichung2
#line1 = [(93,124), (162, 107)]
#line2 = [(107,166), (201, 144)]
#line3 = [(266,159), (340, 140)]
#line4 = [(214,123), (268, 111)]

#line1 = [(300,318), (473, 319)]
#line2 = [(134,439), (440, 450)]
#line3 = [(530,393), (778, 395)]
#line4 = [(520,302), (662, 302)]
#line1 = [(379,397), (589, 402)]
#line2 = [(168,549), (550, 562)]
#line3 = [(662,493), (972, 497)]
#line4 = [(650,378), (827, 378)]
counter1 = 0
counter2 = 0
counter3 = 0
counter4 = 0
counter5 = 0
track_counter1 = 0
track_counter2 = 0
track_counter3 = 0
track_counter4 = 0
track_counter5 = 0
total_km = 0.0
car_total_km = 0.0
small_truck_total_km = 0.0
bus_total_km = 0.0
big_truck_total_km = 0.0
scooter_total_km = 0.0
frame_count = 0
pre_frame_count = 0
labels_to_names = {0:'car',1:'small-truck',2:'bus',3:'big-truck',4:'scooter'}
labels_to_newnames = {0:'sedan',1:'light-truck',2:'bus',3:'heavy-truck',4:'scooter'}
names_to_labels = {'car':0,'small-truck':1,'bus':2,'big-truck':3,'scooter':4}
from collections import deque
pts = [deque(maxlen=30) for _ in range(9999)]
import warnings
warnings.filterwarnings('ignore')
# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def main(_argv):
    f = open('sort_info_10_coco1344_benchmark.txt', 'w+')
    memory = {}
    pre_frame_count_map = {}

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    #global frame_num
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            #cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        
        # encode yolo detections and feed to tracker
        #features = encoder(frame, bboxes)
        #detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        detections = [Detection(bbox, score, class_name) for bbox, score, class_name in zip(bboxes, scores, names)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        '''
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        '''
        #print(boxs.tolist())
        #print(list(scores))
        #print(list(classes))
        idxs = cv2.dnn.NMSBoxes(boxs.tolist(), list(scores), float(0.35), float(0.25))
        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxs.tolist()[i][0], boxs.tolist()[i][1])
                (w, h) = (boxs.tolist()[i][2], boxs.tolist()[i][3])
                dets.append([x, y, x+w, y+h, list(scores)[i],names_to_labels[list(classes)[i]]])

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        #print(dets)
        tracks = tracker.update(dets)
        #print(tracks)
        #print(0/0)
        boxes = []
        indexIDs = []
        #class_tmp = []
        c = []
        previous = memory.copy()
        memory = {}
        # update tracks
        for track in tracks:
            print(track)
            boxes.append([track[0], track[1], track[2], track[3],int(track[5])])
            indexIDs.append(int(track[4]))
            #class_tmp.append(track[5])
            memory[indexIDs[-1]] = boxes[-1]
            #2021/07/05
            color = [int(c) for c in COLORS[int(track[4]) % len(COLORS)]]
            cv2.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), color, 2)
            #cv2.rectangle(frame, (int(track[0]), int(track[1]-30)), (int(track[0])+(len(boxes)+len(indexIDs))*17, int(track[1])), color, -1)
            cv2.rectangle(frame, (int(track[0]), int(track[1]-30)), (int(track[0])+(len(labels_to_newnames[int(track[5])]))*10, int(track[1])), color, -1)
            text = "{}".format(labels_to_newnames[int(track[5])])
            cv2.putText(frame, text, (int(track[0]), int(track[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            #cv2.putText(frame, text, (int(track[0]), int(track[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))
    
                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                #2021/07/02
                #color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                #cv2.rectangle(frame, (x, y), (w, h), color, 2)
    
                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                    #cv2.line(frame, p0, p1, color, 3)
                    global counter1
                    global counter2
                    global counter3
                    global counter4
                    global counter5
                    global track_counter1
                    global track_counter2
                    global track_counter3
                    global track_counter4
                    global track_counter5
                    global total_km
                    global car_total_km
                    global small_truck_total_km
                    global bus_total_km
                    global big_truck_total_km
                    global scooter_total_km

                    if intersect(p0, p1, line1[0], line1[1]) and labels_to_names[box[4]] == 'car': #class_names[i] classes[i]
                        pre_frame_count_map[indexIDs[i]] = frame_count
                        counter1 += 1
                    elif intersect(p0, p1, line1[0], line1[1]) and labels_to_names[box[4]] == 'small-truck':
                        pre_frame_count_map[indexIDs[i]] = frame_count
                        counter2 += 1
                    elif intersect(p0, p1, line1[0], line1[1]) and labels_to_names[box[4]] == 'bus':
                        pre_frame_count_map[indexIDs[i]] = frame_count
                        counter3 += 1
                    elif intersect(p0, p1, line1[0], line1[1]) and labels_to_names[box[4]] == 'big-truck':
                        pre_frame_count_map[indexIDs[i]] = frame_count
                        counter4 += 1
                    elif intersect(p0, p1, line1[0], line1[1]) and labels_to_names[box[4]] == 'scooter':
                        pre_frame_count_map[indexIDs[i]] = frame_count
                        counter5 += 1
    
                    elif intersect(p0, p1, line2[0], line2[1]) and labels_to_names[box[4]] == 'car':
                        #counter1 += 0.5
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter1 += 1
                            speed = int(10/((frame_count-pre_frame_count_map[indexIDs[i]])/30)*3600/1000)
                            total_km = total_km + speed
                            car_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)
                            f.write(str(indexIDs[i]))
                            f.write(': ')
                            f.write(text_speed)
                            f.write('\n')
                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2)#get_color
                    elif intersect(p0, p1, line2[0], line2[1]) and labels_to_names[box[4]] == 'small-truck':
                        #counter2 += 0.5
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter2 += 1
                            speed = int(10/((frame_count-pre_frame_count_map[indexIDs[i]])/30)*3600/1000)
                            total_km = total_km + speed
                            small_truck_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)
                            f.write(str(indexIDs[i]))
                            f.write(': ')
                            f.write(text_speed)
                            f.write('\n')
                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2)#get_color
                    elif intersect(p0, p1, line2[0], line2[1]) and labels_to_names[box[4]] == 'bus':
                        #counter3 += 0.5
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter3 += 1
                            speed = int(10/((frame_count-pre_frame_count_map[indexIDs[i]])/30)*3600/1000)
                            total_km = total_km + speed
                            bus_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)
                            f.write(str(indexIDs[i]))
                            f.write(': ')
                            f.write(text_speed)
                            f.write('\n')
                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2)#get_color
                    elif intersect(p0, p1, line2[0], line2[1]) and labels_to_names[box[4]] == 'big-truck':
                        #counter4 += 0.5
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter4 += 1
                            speed = int(10/((frame_count-pre_frame_count_map[indexIDs[i]])/30)*3600/1000)
                            total_km = total_km + speed
                            big_truck_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)
                            f.write(str(indexIDs[i]))
                            f.write(': ')
                            f.write(text_speed)
                            f.write('\n')
                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2)#get_color
                    elif intersect(p0, p1, line2[0], line2[1]) and labels_to_names[box[4]] == 'scooter':
                        #counter5 += 0.5
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter5 += 1
                            speed = int(10/((frame_count-pre_frame_count_map[indexIDs[i]])/30)*3600/1000)
                            total_km = total_km + speed
                            scooter_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)
                            f.write(str(indexIDs[i]))
                            f.write(': ')
                            f.write(text_speed)
                            f.write('\n')
                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2)#get_color
    
                    elif intersect(p0, p1, line3[0], line3[1]) and labels_to_names[box[4]] == 'car':
                        pre_frame_count_map[indexIDs[i]] = frame_count
                        counter1 += 1
                    elif intersect(p0, p1, line3[0], line3[1]) and labels_to_names[box[4]] == 'small-truck':
                        pre_frame_count_map[indexIDs[i]] = frame_count
                        counter2 += 1
                    elif intersect(p0, p1, line3[0], line3[1]) and labels_to_names[box[4]] == 'bus':
                        pre_frame_count_map[indexIDs[i]] = frame_count
                        counter3 += 1
                    elif intersect(p0, p1, line3[0], line3[1]) and labels_to_names[box[4]] == 'big-truck':
                        pre_frame_count_map[indexIDs[i]] = frame_count
                        counter4 += 1
                    elif intersect(p0, p1, line3[0], line3[1]) and labels_to_names[box[4]] == 'scooter':
                        pre_frame_count_map[indexIDs[i]] = frame_count
                        counter5 += 1
    
                    elif intersect(p0, p1, line4[0], line4[1]) and labels_to_names[box[4]] == 'car':
                        #counter1 += 0.5
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter1 += 1
                            speed = int(10/((frame_count-pre_frame_count_map[indexIDs[i]])/30)*3600/1000)
                            total_km = total_km + speed
                            car_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)
                            f.write(str(indexIDs[i]))
                            f.write(': ')
                            f.write(text_speed)
                            f.write('\n')
                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2)#get_color
                    elif intersect(p0, p1, line4[0], line4[1]) and labels_to_names[box[4]] == 'small-truck':
                        #counter2 += 0.5
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter2 += 1
                            speed = int(10/((frame_count-pre_frame_count_map[indexIDs[i]])/30)*3600/1000)
                            total_km = total_km + speed
                            small_truck_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)
                            f.write(str(indexIDs[i]))
                            f.write(': ')
                            f.write(text_speed)
                            f.write('\n')
                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2)#get_color
                    elif intersect(p0, p1, line4[0], line4[1]) and labels_to_names[box[4]] == 'bus':
                        #counter3 += 0.5
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter3 += 1
                            speed = int(10/((frame_count-pre_frame_count_map[indexIDs[i]])/30)*3600/1000)
                            total_km = total_km + speed
                            bus_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)
                            f.write(str(indexIDs[i]))
                            f.write(': ')
                            f.write(text_speed)
                            f.write('\n')
                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2)#get_color
                    elif intersect(p0, p1, line4[0], line4[1]) and labels_to_names[box[4]] == 'big-truck':
                        #counter4 += 0.5
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter4 += 1
                            speed = int(10/((frame_count-pre_frame_count_map[indexIDs[i]])/30)*3600/1000)
                            total_km = total_km + speed
                            big_truck_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)
                            f.write(str(indexIDs[i]))
                            f.write(': ')
                            f.write(text_speed)
                            f.write('\n')
                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2)#get_color
                    elif intersect(p0, p1, line4[0], line4[1]) and labels_to_names[box[4]] == 'scooter':
                        #counter5 += 0.5
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter5 += 1
                            speed = int(10/((frame_count-pre_frame_count_map[indexIDs[i]])/30)*3600/1000)
                            total_km = total_km + speed
                            scooter_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)
                            f.write(str(indexIDs[i]))
                            f.write(': ')
                            f.write(text_speed)
                            f.write('\n')
                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2)#get_color
                    else:
                        pass
                    #text = "{}".format(classes[i])
                    #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
                #text = "{}: {:.4f}".format(classes, confidences[i])
                #text = "{}".format(indexIDs[i])
                #2021/07/02
                #text = "{}".format(classes[i])
                #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), labels_to_names[box[4]], (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        #cv2.line(frame, line1[0], line1[1], (192,192,192), 1)
        #cv2.line(frame, line2[0], line2[1], (192,192,192), 1)
        #cv2.line(frame, line3[0], line3[1], (192,192,192), 1)
        #cv2.line(frame, line4[0], line4[1], (192,192,192), 1)
        counter_text = "sedan counter:{}".format(int(counter1))
        cv2.putText(frame, counter_text, (10,90), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 1)
        counter_text = "light-truck counter:{}".format(int(counter2))
        cv2.putText(frame, counter_text, (10,210), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 1)
        counter_text = "bus counter:{}".format(int(counter3))
        cv2.putText(frame, counter_text, (10,170), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 1)
        counter_text = "heavy-truck counter:{}".format(int(counter4))
        cv2.putText(frame, counter_text, (10,50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 1)
        counter_text = "scooter counter:{}".format(int(counter5))
        cv2.putText(frame, counter_text, (10,130), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 1)
        frame_count = frame_num + 1

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    print('total_km: ',int(total_km))
    print('counter: ',int(counter1+counter2+counter3+counter4+counter5))
    print('avg_speed: ',float(int(total_km)/int(counter1+counter2+counter3+counter4+counter5)))
    print('sedan_total_km: ',int(car_total_km))
    print('light_truck_total_km: ',int(small_truck_total_km))
    print('bus_total_km: ',int(bus_total_km))
    print('heavy_truck_total_km: ',int(big_truck_total_km))
    print('scooter_total_km: ',int(scooter_total_km))
    print('sedan_track_counter: ',int(track_counter1))
    print('light_truck_track_counter: ',int(track_counter2))
    print('bus_track_counter: ',int(track_counter3))
    print('heavy_truck_track_counter: ',int(track_counter4))
    print('scooter_track_counter: ',int(track_counter5))
    f.write('total_km: '+str(int(total_km)))
    f.write('\n')
    f.write('counter: '+str(int(counter1+counter2+counter3+counter4+counter5)))
    f.write('\n')
    f.write('avg_speed: '+str(float(int(total_km)/int(counter1+counter2+counter3+counter4+counter5))))
    f.write('\n')
    f.write('sedan_total_km: '+str(int(car_total_km)))
    f.write('\n')
    f.write('light_truck_total_km: '+str(int(small_truck_total_km)))
    f.write('\n')
    f.write('bus_total_km: '+str(int(bus_total_km)))
    f.write('\n')
    f.write('heavy_truck_total_km: '+str(int(big_truck_total_km)))
    f.write('\n')
    f.write('scooter_total_km: '+str(int(scooter_total_km)))
    f.write('\n')
    f.write('sedan_track_counter: '+str(int(track_counter1)))
    f.write('\n')
    f.write('light_truck_track_counter: '+str(int(track_counter2)))
    f.write('\n')
    f.write('bus_track_counter: '+str(int(track_counter3)))
    f.write('\n')
    f.write('heavy_truck_track_counter: '+str(int(track_counter4)))
    f.write('\n')
    f.write('scooter_track_counter: '+str(int(track_counter5)))
    #f.write('\n')
    #f.write('car_avg_km: '+str(int(car_total_km)/int(track_counter1)))
    #f.write('\n')
    #f.write('small_truck_avg_km: '+str(int(small_truck_total_km)/int(track_counter2)))
    #f.write('\n')
    #f.write('bus_avg_km: '+str(int(bus_total_km)/int(track_counter3)))
    #f.write('\n')
    #f.write('big_truck_avg_km: '+str(int(big_truck_total_km)/int(track_counter4)))
    #f.write('\n')
    #f.write('scooter_avg_km: '+str(int(scooter_total_km)/int(track_counter5)))
    f.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
