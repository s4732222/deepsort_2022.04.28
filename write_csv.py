#有包含計算影片長度的write_csv.py
import csv
from apscheduler.schedulers.blocking import BlockingScheduler 
'''
# 開啟輸出的 CSV 檔案
with open('output.csv', 'w', newline='') as csvfile:
# 建立 CSV 檔寫入器
writer = csv.writer(csvfile)

# 寫入一列資料
writer.writerow(['影片數', '實際', '估算'])

# 寫入另外幾列資料
writer.writerow(['1', 175, 60])
writer.writerow(['2', 165, 77])
'''
def job_function(): 
  name = ['sedan','Light-truck','Bus','Truck','Scooter']
  txt_data = []
  all_data = []
  with open("C:\\Users\\ADMIN\\Desktop\\yolov4-deepsort\\deepsort_info_10_coco1344_benchmark.txt") as f: #讀辨識完的資料F:\\Desktop\\a.txt
      lines = len(f.readlines()) 
      for i in range(0,11): #10=5類(車速+車量)
          data = 0
          for j in range(0,lines,11):
              f.seek(0)
              temp = f.readlines()[i]
              txt_data.append(float(temp))#
              data+=(float(temp))
          all_data.append(data)
  print(txt_data)#未加總 影片1,影片2,影片3...
  print(all_data)#已加總    all_data[-1]=>影片總長度 秒
  
  total_count=all_data[5]+all_data[6]+all_data[7]+all_data[8]+all_data[9]
  # 開啟輸出的 CSV 檔案
  f = open('output.csv', 'w', newline='') #F:\\Desktop\\
  # 建立 CSV 檔寫入器
  writer = csv.writer(f)
  writer.writerow(['車種', '車速', '實際車量', '估算車量', '比例', '影片1', '影片2', '影片3', '手算車量', '第二條線'])
  
'''
  for i in range(5):
      if(all_data[i+5] == 0):
          writer.writerow([name[i], 0, all_data[i+5], all_data[i+5]*3600/all_data[-1], all_data[i+5]/total_count, txt_data[15+i*3], txt_data[16+i*3], txt_data[17+i*3]]) #3600/實際幾秒 20 21 22(4個影片)		  	  
            
          if(all_data[-1]==0):
              writer.writerow([name[i], 0, all_data[i+5], 0, all_data[i+5]/total_count, txt_data[15+i*3], txt_data[16+i*3], txt_data[17+i*3]]) #3600/實際幾秒 20 21 22(4個影片)		  		  
          elif(total_count==0):
              writer.writerow([name[i], 0, all_data[i+5], all_data[i+5]*3600/all_data[-1], 0, txt_data[15+i*3], txt_data[16+i*3], txt_data[17+i*3]]) #3600/實際幾秒 20 21 22(4個影片)		  		  
          elif(all_data[-1]==0 and total_count==0):		  
              writer.writerow([name[i], 0, all_data[i+5], 0, 0, txt_data[15+i*3], txt_data[16+i*3], txt_data[17+i*3]]) #3600/實際幾秒 20 21 22(4個影片)		  		  
          else: 			  
              writer.writerow([name[i], 0, all_data[i+5], all_data[i+5]*3600/all_data[-1], all_data[i+5]/total_count, txt_data[15+i*3], txt_data[16+i*3], txt_data[17+i*3]]) #3600/實際幾秒 20 21 22(4個影片)		  
          		  
      else:
          writer.writerow([name[i], all_data[i]/all_data[i+5], all_data[i+5], all_data[i+5]*3600/all_data[-1], all_data[i+5]/total_count, txt_data[15+i*3], txt_data[16+i*3], txt_data[17+i*3]]) #'車種', '車速', '實際', '估算'
'''

job_function()
'''
if __name__ == '__main__':
  #job_function()

  sched = BlockingScheduler() 
  # Schedules job_function to be run on the third Friday 
  # of June, July, August, November and December at 00:00, 01:00, 02:00 and 03:00 
  #sched.add_job(job_function, 'cron', year=2021,month = 11,day = 1,hour = 15,minute = 51,second = '3/5') #, 'interval', seconds=5)
  sched.add_job(job_function, 'date', run_date='2022-04-13 14:00:00', args=[])
  #sched.add_job(main, 'interval', seconds=5)
  sched.start() 
'''
