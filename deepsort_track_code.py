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
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from apscheduler.schedulers.blocking import BlockingScheduler

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-608',
                    'path to weights file')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/Scooter.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.5, 'iou threshold') #0.5
flags.DEFINE_float('score', 0.25, 'score threshold') #0.25
flags.DEFINE_boolean('dont_show', True, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', True, 'count objects being tracked on screen')
from collections import deque #Trajectory
pts = [deque(maxlen=30) for _ in range(9999)] #Trajectory 最多保留30個點連線 id可以有9999個
import math

#台灣大道惠中路-最終版本
line1 = [(100,625), (841,616)]
line2 = [(2,690), (834,668)]
line3 = [(977,637), (1850,662)]
line4 = [(986,711), (1950,717)]
'''
#台灣大道惠中路
line1 = [(2,735), (834,703)]
line2 = [(2,846), (851,810)]
line3 = [(0,0), (0,0)]
line4 = [(0,0), (0,0)]
'''

'''
#中清路(曉明女中)-最終版本
line1 = [(278,589), (894,563)]
line2 = [(103,695), (876,665)]
line3 = [(885,513), (1490,521)]
line4 = [(888,574), (1575,578)]
'''
'''
#中清路車流比例
line1 = [(348,512), (868,487)]
line2 = [(316,542), (863,510)]
line3 = [(930,495), (1560,497)]
line4 = [(930,522), (1631,534)]
'''

'''
#台灣大道&朝富路口即時車流
line1 = [(400,320), (890,320)]
line2 = [(300,378), (890,378)]
line3 = [(1025,456), (1895,456)]
line4 = [(918,577), (1895,577)]
'''
'''
#潭子 4m
line1 = [(445,801), (1110,801)]
line2 = [(430,883), (1117,883)]
line3 = [(0,0), (0,0)]
line4 = [(0,0), (0,0)]
'''
'''
#潭子 4m
line1 = [(554,770), (1095,770)]
line2 = [(554,863), (1095,863)]
line3 = [(0,0), (0,0)]
line4 = [(0,0), (0,0)]
'''

'''
#179 10
line1 = [(32,113), (136,141)]
line2 = [(1,137), (89,168)]
line3 = [(149,145), (283,172)]
line4 = [(180,124), (285,142)]
'''
'''
#Daya 10
line1 = [(82,286), (326,315)]
line2 = [(9,305), (264,341)]
line3 = [(307,332), (673,393)]
line4 = [(387,315), (674,358)]
'''

'''
#benchmark
#line1 = [(504,554), (1009,553)]
line1 = [(499,576), (1028,581)]
line2 = [(430,662), (1077,664)]
line3 = [(1143,613), (1808,639)]
#line4 = [(1062,526), (1785,567)]
line4 = [(1087,550), (1810,581)]
'''
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
#http://117.56.11.141:8601/Interface/Cameras/GetJPEGStream?Camera=C503&Width=352&Height=240&Quality=100&FPS=60&AuthUser=web
'''
line1 = [(0,0), (0,0)]
line2 = [(0,0), (0,0)]
line3 = [(0,126), (291,126)]
line4 = [(44,106), (291,106)]
'''
'''
#new179 https://cctvc.freeway.gov.tw/abs2mjpg/bmjpg?camera=214
line1 = [(1,117), (150,117)]
line2 = [(1,135), (125,135)]
line3 = [(100,160), (268,160)]
line4 = [(147,135), (271,135)]
'''
'''
#benchmark
#line1 = [(504,554), (1009,553)]
line1 = [(499,576), (1028,581)]
line2 = [(430,662), (1077,664)]
line3 = [(1143,613), (1808,639)]
#line4 = [(1062,526), (1785,567)]
line4 = [(1087,550), (1810,581)]
'''
'''
line1 = [(0,0), (0,0)]
line2 = [(0,0), (0,0)]
line3 = [(234,154), (430,162)]
line4 = [(248,115), (420,115)]
line3 = [(234,153), (430,162)]
line4 = [(248,112), (420,120)]
'''
'''
#huwei1-3
line1 = [(542,517), (879,517)]
line2 = [(419,652), (880,652)]
line3 = [(892,695), (1360,695)]
line4 = [(889,543), (1253,543)]
'''

'''
#惠來路口 1080p 1920x1080
line1 = [(0,0), (0,0)]
line2 = [(0,0), (0,0)]
line3 = [(866,594), (1820,594)]
line4 = [(888,446), (1576,446)]
'''
'''
#惠來路口 240p 432x240
line1 = [(0,0), (0,0)]
line2 = [(0,0), (0,0)]
line3 = [(195,132), (410,132)]
line4 = [(200,99), (355,99)]
'''

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
Sedan_total_km = 0.0
Light_truck_total_km = 0.0
Bus_total_km = 0.0
Truck_total_km = 0.0
Scooter_total_km = 0.0
frame_count = 0
pre_frame_count = 0
labels_to_names = {0:'Sedan',1:'Light-truck',2:'Bus',3:'Truck',4:'Scooter'}
labels_to_newnames = {0:'sedan',1:'light-truck',2:'Bus',3:'heavy-truck',4:'Scooter'}
names_to_labels = {'Sedan':0,'Light-truck':1,'Bus':2,'Truck':3,'Scooter':4}
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")
import colorsys
GOLDEN_RATIO = 0.618033988749895
def get_color(idx, s=0.8, vmin=0.7):
    h = np.fmod(idx * GOLDEN_RATIO, 1.)
    v = 1. - np.fmod(idx * GOLDEN_RATIO, 1. - vmin)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(255 * b), int(255 * g), int(255 * r)
'''
def Scheduler(_argv):
  sched = BlockingScheduler() 
  # Schedules job_function to be run on the third Friday 
  # of June, July, August, November and December at 00:00, 01:00, 02:00 and 03:00 
  sched.add_job(main, 'date', run_date='2022-04-13 13:43:00', args=[])
  sched.add_job(main, 'date', run_date='2022-04-13 13:45:00', args=[])
  sched.add_job(main, 'date', run_date='2022-04-13 13:48:00', args=[])
  #sched.add_job(main, 'cron', year=2021,month = 11,day = 1,hour = 15,minute = 32,second = 00) #, 'interval', seconds=5)
  #sched.add_job(main, 'interval', seconds=5)
  sched.start() 
# Return true if line segments AB and CD intersect
'''

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def main(_argv): #def main():
    f = open('deepsort_info_10_coco1344_中清路.txt', 'a+')
    memory = {}
    pre_frame_count_map = {}
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/veri.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

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
        #rate = vid.get(5)
        #frame_no = vid.get(7)
        #duration = 	frame_no/rate
    except:
        vid = cv2.VideoCapture(video_path)
    rate = vid.get(5)
    frame_no = vid.get(7)
    duration = 	frame_no/rate
    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter("C:\\Users\\ADMIN\\Desktop\\yolov4-deepsort\\outputs\\outputs_"+time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())+'.avi', codec, fps, (width, height))
        print("success")
        #out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    id_cnt_flag	= {}
    for i in range(9999):	
        id_cnt_flag[i]=False	
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
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        boxes = []
        indexIDs = []
        previous = memory.copy()
        memory = {}
        #id_cnt_flag = {}
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(names_to_labels[class_name]) % len(colors)] #id->class track.track_id->names_to_labels[class_name]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), get_color(int(names_to_labels[class_name])), 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name))*17, int(bbox[1])), get_color(int(names_to_labels[class_name])), -1)
            cv2.putText(frame, labels_to_newnames[names_to_labels[class_name]],(int(bbox[0]), int(bbox[1]-10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
            #cv2.putText(frame, labels_to_newnames[names_to_labels[class_name]] + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            #cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3],class_name])
            indexIDs.append(int(track.track_id))
            #id_cnt_flag[int(track.track_id)]=False			
            memory[indexIDs[-1]] = boxes[-1]
            center = ( int((bbox[0]*2 + bbox[2]-bbox[0])/2), int((bbox[1]*2 + bbox[3]-bbox[1])/2) ) #Trajectory
            pts[int(track.track_id)].append(center) #Trajectory 紀錄當下id及中心點
            #global counter1
            #global counter2
            #global counter3
            #global counter4
            #global counter5			
            for j in range(1, len(pts[int(track.track_id)])): #中心點個數
                if pts[int(track.track_id)][j - 1] is None or pts[int(track.track_id)][j] is None: #前後都有點才可以連線，隔很久只要有點就可以連線
                    continue
                cv2.line(frame,(pts[int(track.track_id)][j-1]), (pts[int(track.track_id)][j]),(get_color(int(names_to_labels[class_name]))),3)			
                """
                if intersect((pts[int(track.track_id)][j-1]), (pts[int(track.track_id)][j]), line2[0], line2[1]) and class_name == 'Light-truck':				
                    counter1 += 1
                elif intersect((pts[int(track.track_id)][j-1]), (pts[int(track.track_id)][j]), line2[0], line2[1]) and class_name == 'Sedan':
                    counter2 += 1				
                elif intersect((pts[int(track.track_id)][j-1]), (pts[int(track.track_id)][j]), line2[0], line2[1]) and class_name == 'Scooter':
                    counter3 += 1				
                elif intersect((pts[int(track.track_id)][j-1]), (pts[int(track.track_id)][j]), line2[0], line2[1]) and class_name == 'Bus':
                    counter4 += 1				
                elif intersect((pts[int(track.track_id)][j-1]), (pts[int(track.track_id)][j]), line2[0], line2[1]) and class_name == 'Light-truck':				
                    counter5 += 1	
                elif intersect((pts[int(track.track_id)][j-1]), (pts[int(track.track_id)][j]), line4[0], line4[1]) and class_name == 'Light-truck':				
                    counter1 += 1
                elif intersect((pts[int(track.track_id)][j-1]), (pts[int(track.track_id)][j]), line4[0], line4[1]) and class_name == 'Sedan':
                    counter2 += 1				
                elif intersect((pts[int(track.track_id)][j-1]), (pts[int(track.track_id)][j]), line4[0], line4[1]) and class_name == 'Scooter':
                    counter3 += 1				
                elif intersect((pts[int(track.track_id)][j-1]), (pts[int(track.track_id)][j]), line4[0], line4[1]) and class_name == 'Bus':
                    counter4 += 1				
                elif intersect((pts[int(track.track_id)][j-1]), (pts[int(track.track_id)][j]), line4[0], line4[1]) and class_name == 'Light-truck':				
                    counter5 += 1						
                """		
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:

                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]-box[0]), int(box[3]-box[1]))
                #print(x,y,w,h)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]-previous_box[0]), int(previous_box[3]-previous_box[1]))
                    #print(x2,y2,w2,h2)
                    #p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    #p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                    p0 = (int((x*2 + w)/2), int((y*2 + h)/2))
                    p1 = (int((x2*2 + w2)/2), int((y2*2 + h2)/2))
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
                    global Sedan_total_km
                    global Light_truck_total_km
                    global Bus_total_km
                    global Truck_total_km
                    global Scooter_total_km

                    if intersect(p0, p1, line1[0], line1[1]) and box[4] == 'Truck':
                        pre_frame_count_map[indexIDs[i]] = frame_num
                        #counter1 += 1
                    elif intersect(p0, p1, line1[0], line1[1]) and box[4] == 'Sedan':
                        pre_frame_count_map[indexIDs[i]] = frame_num
                        #counter2 += 1
                    elif intersect(p0, p1, line1[0], line1[1]) and box[4] == 'Scooter':
                        pre_frame_count_map[indexIDs[i]] = frame_num
                        #counter3 += 1
                    elif intersect(p0, p1, line1[0], line1[1]) and box[4] == 'Bus':
                        pre_frame_count_map[indexIDs[i]] = frame_num
                        #counter4 += 1
                    elif intersect(p0, p1, line1[0], line1[1]) and box[4] == 'Light-truck':
                        pre_frame_count_map[indexIDs[i]] = frame_num
                        #counter5 += 1
    
                    elif intersect(p0, p1, line2[0], line2[1]) and box[4] == 'Truck': #track.get_class()
                        counter1 += 1
                        id_cnt_flag[indexIDs[i]] = True
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter1 += 1
                            speed = int(10/((frame_num-pre_frame_count_map[indexIDs[i]])/fps)*3600/1000)
                            total_km = total_km + speed
                            Truck_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)

                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
                    elif intersect(p0, p1, line2[0], line2[1]) and box[4] == 'Sedan':
                        counter2 += 1		
                        id_cnt_flag[indexIDs[i]] = True						
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter2 += 1
                            speed = int(10/((frame_num-pre_frame_count_map[indexIDs[i]])/fps)*3600/1000)
                            total_km = total_km + speed
                            Sedan_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)

                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    
                    elif intersect(p0, p1, line2[0], line2[1]) and box[4] == 'Scooter':
                        counter3 += 1				
                        id_cnt_flag[indexIDs[i]] = True						
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter3 += 1
                            speed = int(10/((frame_num-pre_frame_count_map[indexIDs[i]])/fps)*3600/1000)
                            total_km = total_km + speed
                            Scooter_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)

                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    
                    elif intersect(p0, p1, line2[0], line2[1]) and box[4] == 'Bus':
                        counter4 += 1				
                        id_cnt_flag[indexIDs[i]] = True						
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter4 += 1
                            speed = int(10/((frame_num-pre_frame_count_map[indexIDs[i]])/fps)*3600/1000)
                            total_km = total_km + speed
                            Bus_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)

                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    
                    elif intersect(p0, p1, line2[0], line2[1]) and box[4] == 'Light-truck':
                        counter5 += 1			
                        id_cnt_flag[indexIDs[i]] = True						
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter5 += 1
                            speed = int(10/((frame_num-pre_frame_count_map[indexIDs[i]])/fps)*3600/1000)
                            total_km = total_km + speed
                            Light_truck_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)

                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    
                    elif intersect(p0, p1, line3[0], line3[1]) and box[4] == 'Truck':
                        pre_frame_count_map[indexIDs[i]] = frame_num
                        #counter1 += 1
                    elif intersect(p0, p1, line3[0], line3[1]) and box[4] == 'Sedan':
                        pre_frame_count_map[indexIDs[i]] = frame_num
                        print(pre_frame_count_map)
                        #counter2 += 1
                    elif intersect(p0, p1, line3[0], line3[1]) and box[4] == 'Scooter':
                        pre_frame_count_map[indexIDs[i]] = frame_num
                        #counter3 += 1
                    elif intersect(p0, p1, line3[0], line3[1]) and box[4] == 'Bus':
                        pre_frame_count_map[indexIDs[i]] = frame_num
                        #counter4 += 1
                    elif intersect(p0, p1, line3[0], line3[1]) and box[4] == 'Light-truck':
                        pre_frame_count_map[indexIDs[i]] = frame_num
                        #counter5 += 1
    
                    elif intersect(p0, p1, line4[0], line4[1]) and box[4] == 'Truck':
                        counter1 += 1	
                        id_cnt_flag[indexIDs[i]] = True						
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter1 += 1
                            speed = int(10/((frame_num-pre_frame_count_map[indexIDs[i]])/fps)*3600/1000)
                            total_km = total_km + speed
                            Truck_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)

                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    
                    elif intersect(p0, p1, line4[0], line4[1]) and box[4] == 'Sedan':
                        counter2 += 1
                        id_cnt_flag[indexIDs[i]] = True						
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter2 += 1
                            speed = int(10/((frame_num-pre_frame_count_map[indexIDs[i]])/fps)*3600/1000)
                            total_km = total_km + speed
                            Sedan_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)

                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    
                    elif intersect(p0, p1, line4[0], line4[1]) and box[4] == 'Scooter':
                        counter3 += 1
                        id_cnt_flag[indexIDs[i]] = True						
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter3 += 1
                            speed = int(10/((frame_num-pre_frame_count_map[indexIDs[i]])/fps)*3600/1000)
                            total_km = total_km + speed
                            Scooter_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)

                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    
                    elif intersect(p0, p1, line4[0], line4[1]) and box[4] == 'Bus':
                        counter4 += 1		
                        id_cnt_flag[indexIDs[i]] = True						
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter4 += 1
                            speed = int(10/((frame_num-pre_frame_count_map[indexIDs[i]])/fps)*3600/1000)
                            total_km = total_km + speed
                            Bus_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)

                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    
                    elif intersect(p0, p1, line4[0], line4[1]) and box[4] == 'Light-truck':
                        counter5 += 1	
                        id_cnt_flag[indexIDs[i]] = True						
                        if(indexIDs[i] in pre_frame_count_map):
                            track_counter5 += 1
                            speed = int(10/((frame_num-pre_frame_count_map[indexIDs[i]])/fps)*3600/1000)
                            total_km = total_km + speed
                            Light_truck_total_km += speed
                            text_speed = "{} km".format(speed)
                            print(text_speed)

                            cv2.putText(frame, text_speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)

                    elif intersect(pts[indexIDs[i]][0], p0, line2[0], line2[1]) and box[4] == 'Truck' and id_cnt_flag[indexIDs[i]] == False: #最初找到的點有沒有相交於新點p0
                        counter1 += 1
                        id_cnt_flag[indexIDs[i]] = True						
                    elif intersect(pts[indexIDs[i]][0], p0, line2[0], line2[1]) and box[4] == 'Sedan' and id_cnt_flag[indexIDs[i]] == False: #最初找到的點有沒有相交於新點p0
                        counter2 += 1
                        id_cnt_flag[indexIDs[i]] = True
                    elif intersect(pts[indexIDs[i]][0], p0, line2[0], line2[1]) and box[4] == 'Scooter' and id_cnt_flag[indexIDs[i]] == False: #最初找到的點有沒有相交於新點p0
                        counter3 += 1
                        id_cnt_flag[indexIDs[i]] = True
                    elif intersect(pts[indexIDs[i]][0], p0, line2[0], line2[1]) and box[4] == 'Bus' and id_cnt_flag[indexIDs[i]] == False: #最初找到的點有沒有相交於新點p0
                        counter4 += 1
                        id_cnt_flag[indexIDs[i]] = True
                    elif intersect(pts[indexIDs[i]][0], p0, line2[0], line2[1]) and box[4] == 'Light-truck' and id_cnt_flag[indexIDs[i]] == False: #最初找到的點有沒有相交於新點p0
                        counter5 += 1
                        id_cnt_flag[indexIDs[i]] = True
                    elif intersect(pts[indexIDs[i]][0], p0, line4[0], line4[1]) and box[4] == 'Truck' and id_cnt_flag[indexIDs[i]] == False: #最初找到的點有沒有相交於新點p0
                        counter1 += 1
                        id_cnt_flag[indexIDs[i]] = True
                    elif intersect(pts[indexIDs[i]][0], p0, line4[0], line4[1]) and box[4] == 'Sedan' and id_cnt_flag[indexIDs[i]] == False: #最初找到的點有沒有相交於新點p0
                        counter2 += 1
                        id_cnt_flag[indexIDs[i]] = True
                    elif intersect(pts[indexIDs[i]][0], p0, line4[0], line4[1]) and box[4] == 'Scooter' and id_cnt_flag[indexIDs[i]] == False: #最初找到的點有沒有相交於新點p0
                        counter3 += 1
                        id_cnt_flag[indexIDs[i]] = True
                    elif intersect(pts[indexIDs[i]][0], p0, line4[0], line4[1]) and box[4] == 'Bus' and id_cnt_flag[indexIDs[i]] == False: #最初找到的點有沒有相交於新點p0
                        counter4 += 1
                        id_cnt_flag[indexIDs[i]] = True
                    elif intersect(pts[indexIDs[i]][0], p0, line4[0], line4[1]) and box[4] == 'Light-truck' and id_cnt_flag[indexIDs[i]] == False: #最初找到的點有沒有相交於新點p0
                        counter5 += 1
                        id_cnt_flag[indexIDs[i]] = True						
						
								
                    else:
                        pass                
                
                #text = "{}".format(box[4])
                #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        cv2.line(frame, line1[0], line1[1], (192,192,192), 1)
        cv2.line(frame, line2[0], line2[1], (192,192,192), 1)
        cv2.line(frame, line3[0], line3[1], (192,192,192), 1)
        cv2.line(frame, line4[0], line4[1], (192,192,192), 1)

        counter_text = "heavy-truck counter:{}".format(int(counter1))
        #cv2.putText(frame, counter_text, (10,50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 1)
        cv2.putText(frame, counter_text, (10,25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 1)
        counter_text = "sedan counter:{}".format(int(counter2))
        #cv2.putText(frame, counter_text, (10,90), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 1)
        cv2.putText(frame, counter_text, (10,45), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 1)
        counter_text = "Scooter counter:{}".format(int(counter3))
        #cv2.putText(frame, counter_text, (10,130), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 1)
        cv2.putText(frame, counter_text, (10,65), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 1)
        counter_text = "Bus counter:{}".format(int(counter4))
        #cv2.putText(frame, counter_text, (10,170), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 1)
        cv2.putText(frame, counter_text, (10,85), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 1)
        counter_text = "light-truck counter:{}".format(int(counter5))
        #cv2.putText(frame, counter_text, (10,210), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 1)
        cv2.putText(frame, counter_text, (10,105), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 1)

        #frame_count = frame_num + 1

        # calculate frames per second of running detections
        FPS = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % FPS)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cv2.destroyAllWindows()
    '''
    print('total_km: ',int(total_km))
    print('counter: ',int(counter1+counter2+counter3+counter4+counter5))
    print('avg_speed: ',float(int(total_km)/int(counter1+counter2+counter3+counter4+counter5)))
    print('sedan_total_km: ',int(Sedan_total_km))
    print('light_truck_total_km: ',int(Light_truck_total_km))
    print('Bus_total_km: ',int(Bus_total_km))
    print('heavy_truck_total_km: ',int(Truck_total_km))
    print('Scooter_total_km: ',int(Scooter_total_km))
    print('sedan_track_counter: ',int(track_counter2))
    print('light_truck_track_counter: ',int(track_counter5))
    print('Bus_track_counter: ',int(track_counter4))
    print('heavy_truck_track_counter: ',int(track_counter1))
    print('Scooter_track_counter: ',int(track_counter3))
    '''

    f.write('total_km: '+str(int(total_km)))
    f.write('\n')
    f.write('counter: '+str(int(counter1+counter2+counter3+counter4+counter5)))
    f.write('\n')
    f.write('avg_speed: '+str(float(int(total_km)/int(counter1+counter2+counter3+counter4+counter5))))
    f.write('\n')
    f.write('Sedan_total_km: ' +str(int(Sedan_total_km)))
    f.write('\n')
    f.write('Light_truck_total_km: ' +str(int(Light_truck_total_km)))
    f.write('\n')
    f.write('Bus_total_km: ' +str(int(Bus_total_km)))
    f.write('\n')
    f.write('Truck_total_km: ' +str(int(Truck_total_km)))
    f.write('\n')
    f.write('Scooter_total_km:' +str(int(Scooter_total_km)))
    f.write('\n')
    f.write('track_counter2: ' +str(int(track_counter2)))
    f.write('\n')
    f.write('track_counter5: ' +str(int(track_counter5)))
    f.write('\n')
    f.write('track_counter4: ' +str(int(track_counter4)))
    f.write('\n')
    f.write('track_counter1: ' +str(int(track_counter1)))
    f.write('\n')
    f.write('track_counter3: ' +str(int(track_counter3)))
    f.write('\n')
    f.write('duration: ' +str(int(duration)))
    f.write('\n')
    f.write('sedan_track_counter: '+str(int(track_counter2)))
    f.write('\n')
    f.write('light_truck_track_counter: '+str(int(track_counter5)))
    f.write('\n')
    f.write('Bus_track_counter: '+str(int(track_counter4)))
    f.write('\n')
    f.write('heavy_truck_track_counter: '+str(int(track_counter1)))
    f.write('\n')
    f.write('Scooter_track_counter: '+str(int(track_counter3)))
    #f.write('\n')
    #f.write('Sedan_avg_km: '+str(int(Sedan_total_km)/int(track_counter2)))
    #f.write('\n')
    #f.write('Light_truck_avg_km: '+str(int(Light_truck_total_km)/int(track_counter5)))
    #f.write('\n')
    #f.write('Bus_avg_km: '+str(int(Bus_total_km)/int(track_counter4)))
    #f.write('\n')
    #f.write('Truck_avg_km: '+str(int(Truck_total_km)/int(track_counter1)))
    #f.write('\n')
    #f.write('Scooter_avg_km: '+str(int(Scooter_total_km)/int(track_counter3)))
    f.close()
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
    Sedan_total_km = 0.0
    Light_truck_total_km = 0.0
    Bus_total_km = 0.0
    Truck_total_km = 0.0
    Scooter_total_km = 0.0
    frame_count = 0
    pre_frame_count = 0




if __name__ == '__main__':
    try:
        #app.run(Scheduler)
        app.run(main)
    except SystemExit:
        pass

