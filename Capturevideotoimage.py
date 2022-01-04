import cv2
import numpy as np
import time
import threading
from os.path import dirname, join

from COCO_model import general_mulitpose_model 
from action_model import PosePredictor

class VideoConverter():
    def __init__(self, number=5):
        self.time_F = number
        self.worktime = 0
        self.videotime = 0

    def print_time(self, second):
        minute = second // 60
        second = second % 60
        hour = minute // 60
        minute = minute % 60

        return str(hour)+':'+str(minute)+':'+str(second)

    def judge_result(self, result):    
        # write result to file to analysis accuracy
        result_file = open("predict_result.txt", "a")

        worker = 0
        for i in range(0, result.shape[0]):
            result_file.write(''.join(str(result[i][0]) + "," + str(result[i][1]) + "\n"))
            # print(result[i][0], " " ,result[i][1])
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
        video = cv2.VideoCapture(filename)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        print("[INFO]Video's Fps:",fps)

        tool = general_mulitpose_model()    # coco model
        print("[Time]Time Taken in Model Loading: {}".format(time.time() - start))

        # f_csv = open("./trans_to_train/test.csv", "w", newline='') # 測試用 到時候真正寫入檔案用追加方式(a+)
        print("[Time]Time Open Csv: {}".format(time.time() - start))
        f_csv = open("./trans_to_train/pose_data.csv", "a", newline='') # 真正寫入檔案

        # read video
        while True:
            ret = video.grab()
            # if frame is read correctly ret is True
            if not ret:
                print("[INFO]Can't receive frame (stream end?). Exiting ...")
                break

            # 
            if(turn % self.time_F == 0):    #5
                ret, frame = video.retrieve() 
                if ret:
                    if(turn % (self.time_F*3) == 0):    
                        train_images.append(frame)
                        dataset = tool.gen_train_dataset(train_images, action)
                        print("[Time]Time Generate Dataset: {}".format(time.time() - start))                   
                        print(np.array(dataset).shape)
                        # print(np.array(dataset))
                        if dataset: 
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
        video.release()   

    # predict video
    def transform_and_predict(self, filename):
        start = time.time()
        turn = 1
        predict_images = []
        video = cv2.VideoCapture(filename)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        print("[INFO]Video's Fps:",fps)
        tool = general_mulitpose_model()
        print("[Time]Time Taken in Model Loading: {}".format(time.time() - start))

        predicter = PosePredictor()
        lstm_model = predicter.load_lstm_model(join(dirname(__file__), './model/lstm_fishman_action.h5'))
        # lstm_model = predicter.load_lstm_model('./model/lstm_fishman_action.h5')
        print("[Time]Time Taken in LSTM Loading: {}".format(time.time() - start))

             
        rest_skip, work_skip, worktime, videotime = [0, 0, 0, 0]   # skip number

        # read video
        while True:
            ret = video.grab()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if(turn % self.time_F == 0 and rest_skip == 0 and work_skip == 0): #5
                print('[Time]'+self.print_time(worktime)+'/'+self.print_time(videotime))
                # self.print_time(worktime)
                # self.print_time(videotime)
                ret, frame = video.retrieve() 
                if ret:
                    if(turn % (self.time_F*3) == 0):    #15 
                        predict_images.append(frame)
                        dataset = tool.gen_predict_dataset(predict_images)  # generate dataset

                        if(dataset == 1):
                            work_skip = 120*fps
                            worktime = worktime + 120
                        elif(dataset == 0):
                            rest_skip = 15*fps  # skip
                        elif(dataset):    
                            dataset = np.array(dataset) # numpy type
                            result = predicter.predict(dataset, lstm_model)  # result

                            if(self.judge_result(result) == True):  # judge result
                                work_skip = 120*fps
                                worktime = worktime + 120
                            else: 
                                rest_skip = 30*fps  # skip
                        else:   
                            rest_skip = 15*fps  # skip

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
            videotime = turn // fps
        # end While

        video.release()
        return worktime, videotime

    # test predict video
    def test_predict(self, filename, rest_skiptime, work_skiptime):
        start = time.time()
        turn = 1
        predict_images = []
        video = cv2.VideoCapture(filename)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        print("[INFO]Video's Fps:",fps)
        tool = general_mulitpose_model()
        print("[Time]Time Taken in Model Loading: {}".format(time.time() - start))

        predicter = PosePredictor()
        # lstm_model = predicter.load_lstm_model('./model/lstm_fishman_action.h5')
        lstm_model = predicter.load_lstm_model(join(dirname(__file__), './model/lstm_fishman_action.h5'))
        print("[Time]Time Taken in LSTM Loading: {}".format(time.time() - start))
       
        rest_skip, work_skip, worktime, videotime = [0, 0, 0, 0]   # skip number

        # read video
        while True:
            ret = video.grab()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if(turn % self.time_F == 0 and rest_skip == 0 and work_skip == 0): #5
                print('[Time]'+self.print_time(worktime)+'/'+self.print_time(videotime))
                # self.print_time(worktime)
                # self.print_time(videotime)
                ret, frame = video.retrieve() 
                if ret:
                    if(turn % (self.time_F*3) == 0):    #15 
                        predict_images.append(frame)
                        dataset = tool.gen_predict_dataset(predict_images)  # generate dataset

                        if(dataset == 1):
                            work_skip = work_skiptime*fps
                            worktime = worktime + work_skiptime
                        elif(dataset == 0):
                            rest_skip = rest_skiptime*fps   # skip
                        elif(dataset):    
                            dataset = np.array(dataset) # numpy type
                            result = predicter.predict(dataset, lstm_model)  # result

                            if(self.judge_result(result) == True):  
                                work_skip = work_skiptime*fps
                                worktime = worktime + work_skiptime
                            else: 
                                rest_skip = rest_skiptime*fps   # skip
                        else:   
                            rest_skip = rest_skiptime*fps   # skip

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
            videotime = turn // fps
        # end While

        video.release()
        return worktime, videotime 

if __name__ == '__main__':
    vc = VideoConverter()
    # vc.transform_to_traindata(join(dirname(__file__), "./train_images/12-30/rest/Project 2-1.avi"), ["rest"])
    # vc.transform_and_predict(join(dirname(__file__), "./train_images/longtest.mp4"))
    vc.test_predict(join(dirname(__file__), "./train_images/longtest.mp4"), 30, 120)
    cv2.destroyAllWindows()


