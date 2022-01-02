import cv2
import numpy as np
import time
import threading
from os.path import dirname, join

from COCO_model import general_mulitpose_model 
from action_model import PosePredictor

class VedioConverter():
    def __init__(self, number=5):
        self.time_F = number
        self.worktime = 0
        self.vediotime = 0

    def print_time(self, second):
        minute = second // 60
        second = second % 60
        hour = minute // 60
        minute = minute % 60

        print("[Time]{}:{}:{}" .format(hour, minute, second))

    def judge_result(self, result):    
        # write result to file to analysis accuracy
        result_file = open("predict_result.txt", "a")

        worker = 0
        for i in range(0, result.shape[0]):
            result_file.write(''.join(str(result[i][0]) + "," + str(result[i][1]) + "\n"))
            print(result[i][0], " " ,result[i][1])
            if( result[i][0] > result[i][1]):
                worker = worker + 1

        result_file.close()
        if( worker/result.shape[0] > 0.5 ): #工作者/總人數
            return True
        else:
            return False


    # 一次把整部影片轉成圖片 並產生訓練用csv檔            
    def transform_to_traindata(self, filename, action):
        start = time.time()
        turn = 1
        train_images = []
        vedio = cv2.VideoCapture(filename)
        fps = int(vedio.get(cv2.CAP_PROP_FPS))
        print("[INFO]Vedio's Fps:",fps)

        tool = general_mulitpose_model()    # coco model
        print("[Time]Time Taken in Model Loading: {}".format(time.time() - start))

        # f_csv = open("./trans_to_train/test.csv", "w", newline='') # 測試用 到時候真正寫入檔案用追加方式(a+)
        print("[Time]Time Open Csv: {}".format(time.time() - start))
        f_csv = open("./trans_to_train/pose_data.csv", "a", newline='') # 真正寫入檔案

        # 讀影片
        while True:
            ret = vedio.grab()
            # if frame is read correctly ret is True
            if not ret:
                print("[INFO]Can't receive frame (stream end?). Exiting ...")
                break

            # 每隔幾幀進行擷取 省略每次都需read()的時間 改用grab + retrieve
            if(turn % self.time_F == 0):    #5
                ret, frame = vedio.retrieve() 
                if ret:
                    if(turn % (self.time_F*3) == 0):    #15 表示已經有3張圖了
                        train_images.append(frame)
                        dataset = tool.gen_train_dataset(train_images, action)
                        print("[Time]Time Generate Dataset: {}".format(time.time() - start))                   
                        print(np.array(dataset).shape)
                        # print(np.array(dataset))
                        if dataset: # 有數據 表示是正常可用的 
                            tool.gen_train_csv(dataset, f_csv)  
                                
                        turn = 1
                        train_images.clear()  
                    else:
                        train_images.append(frame)
                else:
                    print("[INFO]Can't receive frame (stream end?). Exiting ...") 
                    break    
            turn = turn + 1
        # end while
        f_csv.close()
        vedio.release()   

    # 把影片每3張圖 做一次預測資料 (預測用)
    def transform_and_predict(self, filename):
        start = time.time()
        turn = 1
        predict_images = []
        vedio = cv2.VideoCapture(filename)
        fps = int(vedio.get(cv2.CAP_PROP_FPS))
        print("[INFO]Vedio's Fps:",fps)
        tool = general_mulitpose_model()
        print("[Time]Time Taken in Model Loading: {}".format(time.time() - start))

        predicter = PosePredictor()
        lstm_model = predicter.load_lstm_model('./model/lstm_fishman_action.h5')
        print("[Time]Time Taken in LSTM Loading: {}".format(time.time() - start))

        
        rest_skip, work_skip, worktime, vediotime = [0, 0, 0, 0]   # skip代表要跳過的幀數 time代表紀錄時間
   
        while True:
            ret = vedio.grab()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # 每n幀做擷取 每3n幀做擷取並把圖片集拿去預測一次 
            if(turn % self.time_F == 0 and rest_skip == 0 and work_skip == 0): #5
                self.print_time(worktime)
                self.print_time(vediotime)
                ret, frame = vedio.retrieve() 
                if ret:
                    if(turn % (self.time_F*3) == 0):    #15 表示已經有3張圖了
                        predict_images.append(frame)
                        dataset = tool.gen_predict_dataset(predict_images)  # 圖片轉成dataset

                        # dataset return  1 = 人數夠多  0 = 人數過少
                        if(dataset == 1):
                            work_skip = 120*fps
                            worktime = worktime + 120
                        elif(dataset == 0):
                            rest_skip = 15*fps
                        elif(dataset):    # 有可用資料
                            dataset = np.array(dataset) # 轉numpy type
                            result = predicter.predict(dataset, lstm_model)  # 預測結果

                            if(self.judge_result(result) == True):  # 有工作狀態->跳120秒 沒有->跳15秒
                                work_skip = 120*fps
                                worktime = worktime + 120
                            else: 
                                rest_skip = 15*fps
                        else:   #　沒資料直接跳過
                            rest_skip = 15*fps

                        predict_images.clear()
                    else:
                        predict_images.append(frame)
                else:
                    print("Can't receive frame (stream end?). Exiting ...") 
                    break
            elif(rest_skip > 0):
                rest_skip = rest_skip - 1
            elif(work_skip > 0):
                work_skip = work_skip - 1

            turn = turn + 1
            vediotime = turn // fps
        # end While

        vedio.release()
        return worktime, vediotime

    def test_predict(self, filename, rest_skiptime, work_skiptime):
        start = time.time()
        turn = 1
        predict_images = []
        vedio = cv2.VideoCapture(filename)
        fps = int(vedio.get(cv2.CAP_PROP_FPS))
        print("[INFO]Vedio's Fps:",fps)
        tool = general_mulitpose_model()
        print("[Time]Time Taken in Model Loading: {}".format(time.time() - start))

        predicter = PosePredictor()
        # lstm_model = predicter.load_lstm_model('./model/lstm_fishman_action.h5')
        lstm_model = predicter.load_lstm_model(join(dirname(__file__), './model/lstm_fishman_action.h5'))
        print("[Time]Time Taken in LSTM Loading: {}".format(time.time() - start))

        
        rest_skip, work_skip, worktime, vediotime = [0, 0, 0, 0]   # skip代表要跳過的幀數 time代表紀錄時間
   
        while True:
            ret = vedio.grab()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # 每n幀做擷取 每3n幀做擷取並把圖片集拿去預測一次 
            if(turn % self.time_F == 0 and rest_skip == 0 and work_skip == 0): #5
                self.print_time(worktime)
                self.print_time(vediotime)
                ret, frame = vedio.retrieve() 
                if ret:
                    if(turn % (self.time_F*3) == 0):    #15 表示已經有3張圖了
                        predict_images.append(frame)
                        dataset = tool.gen_predict_dataset(predict_images)  # 圖片轉成dataset

                        if(dataset == 1):
                            work_skip = work_skiptime*fps
                            worktime = worktime + work_skiptime
                        elif(dataset == 0):
                            rest_skip = rest_skiptime*fps
                        elif(dataset):    # 有可用資料
                            dataset = np.array(dataset) # 轉numpy type
                            result = predicter.predict(dataset, lstm_model)  # 預測結果

                            if(self.judge_result(result) == True):  # 有工作狀態->跳120秒 沒有->跳15秒
                                work_skip = work_skiptime*fps
                                worktime = worktime + work_skiptime
                            else: 
                                rest_skip = rest_skiptime*fps
                        else:   #　沒資料直接跳過
                            rest_skip = rest_skiptime*fps

                        predict_images.clear()
                    else:
                        predict_images.append(frame)
                else:
                    print("Can't receive frame (stream end?). Exiting ...") 
                    break
            elif(rest_skip > 0):
                rest_skip = rest_skip - 1
            elif(work_skip > 0):
                work_skip = work_skip - 1

            turn = turn + 1
            vediotime = turn // fps
        # end While

        vedio.release()
        return worktime, vediotime 

if __name__ == '__main__':
    vc = VedioConverter()
    # vc.transform_to_traindata(join(dirname(__file__), "./train_images/12-30/rest/Project 2-1.avi"), ["rest"])
    # vc.transform_and_predict(join(dirname(__file__), "./train_images/longtest.mp4"))
    vc.test_predict(join(dirname(__file__), "./train_images/longtest.mp4"), 15, 120)
    cv2.destroyAllWindows()


