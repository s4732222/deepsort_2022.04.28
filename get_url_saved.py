from __future__ import division, print_function, absolute_import
import cv2
import numpy as np
import requests
import http.client
http.client.HTTPConnection._http_vsn = 10
http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'
import time
from apscheduler.schedulers.blocking import BlockingScheduler 
#import sys 
import gc

def job_function(): 

  global bytes
  writer = None
  #https://cctvc.freeway.gov.tw/abs2mjpg/bmjpg?camera=485
  #https://cctvc.freeway.gov.tw/abs2mjpg/bmjpg?camera=329
  #http://117.56.11.142:7001/media/00-0F-7C-16-C9-AE.mpjpeg?resolution=240p&auth=cHVibGljOjVjNDI3YjY5OGI0ZDg6YzM2M2MzY2Q0ZTY0ZDA4ZjE1MjE0ODZmZjg1ZjQ3NmY
  #https://cctvc.freeway.gov.tw/abs2mjpg/bmjpg?camera=485 大雅 20
  #https://cctvc.freeway.gov.tw/abs2mjpg/bmjpg?camera=214  nantun 15
  #https://thbcctv08.thb.gov.tw/T3-189K+750 #10
  #http://117.56.11.142:7001/media/58:03:fb:5e:22:2a.mpjpeg?resolution=1080p&auth=cHVibGljOjVjNzExMjI2NmQwMDA6MTYxNWVlODYxMjJjYTE2YzI3ZTk5YzE5ZGE0ZDAzNzg C793大里
  #http://117.56.11.142:7001/media/58:03:fb:5e:22:29.mpjpeg?resolution=240p&auth=cHVibGljOjVjZDkzZGQyNDcxYTg6ODUyNTg2ZjcyMmM0ZmYyYzM1NWE5NjM5ZThlMmY5Yjk
  #http://117.56.11.141:8601/Interface/Cameras/GetJPEGStream?Camera=C503&Width=352&Height=240&Quality=100&FPS=60&AuthUser=web #14

  r = requests.get('https://cctvc.freeway.gov.tw/abs2mjpg/bmjpg?camera=329', stream=True)#, auth=('user', 'password') https://cctvc.freeway.gov.tw/abs2mjpg/bmjpg?camera=336&0.04042071613553566

  #t1 = time.time()
  
  # loop over frames from the video file stream
  if(r.status_code == 200):
      bytes = bytes()
      frameIndex = 0
      t1 = time.time()
      for chunk in r.iter_content(chunk_size=1024):
          #frameIndex += 1
          bytes += chunk
          a = bytes.find(b'\xff\xd8')
          b = bytes.find(b'\xff\xd9')
          if a != -1 and b != -1:
              jpg = bytes[a:b+2]
              bytes = bytes[b+2:]
              frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
              frameIndex += 1    
              t2 = time.time()
              print('fps: ',frameIndex/(t2-t1))
              # saves image file
              #cv2.imwrite("output/taichung/frame-{}.png".format(frameIndex), frame)
      
              #cv2.imshow('Taichung transportation counting system (press Q to exit)', frame)
              
              # Press Q to stop!
              if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
              if (t2-t1)>60: #先跑 5s 熱機使fps穩定
                  if writer is None:
                      # initialize our video writer
                      fourcc = cv2.VideoWriter_fourcc(*"XVID")
                      writer = cv2.VideoWriter('F:\\yolov4-deepsort\\video_'+time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())+'.mp4', fourcc, 10, frameIndex/(t2-t1), (frame.shape[1], frame.shape[0]), True) #30 10.5
              
                  # write the output frame to disk
                  writer.write(frame)
  
              if (t2-t1)>185:#35: #test 10s 180
              #if frameIndex >= 4000: # limits the execution to the first 4000 frames
                  print("[INFO] cleaning up...")
                  writer.release()
                  cv2.destroyAllWindows()
                  del bytes				  
                  gc.collect()
                  break
                  #frame.release()
                  #exit()
                  #sys.exit()
              
  else:
      print("Received unexpected status code {}".format(r.status_code))
  # release the file pointers
  #t3 = time.time()
  #print("the video time is "+str(t3-t1)+"s")
  #print("[INFO] cleaning up...")
  #cv2.destroyAllWindows()
  #writer.release()
  print("done")


job_function()

'''
scheduler=BlockingScheduler()
scheduler.add_job(job_function, "interval",minutes=5)
scheduler.start()
'''


'''
if __name__ == '__main__':
  #job_function()

  sched = BlockingScheduler() 
  sched.add_job(job_function, "interval",seconds=10)
  #sched.add_job(job_function, 'cron', year=2021,month = 11,day = 2,hour = 15,minute = '14/15',second = 0) #, 'interval', seconds=5)
  #sched.add_job(job_function, 'date', run_date='2022-04-16 18:56:00', args=[])
  sched.start() 
'''



