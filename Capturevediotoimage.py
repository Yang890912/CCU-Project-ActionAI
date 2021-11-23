import cv2
import numpy as np
import time

from COCO_model import general_mulitpose_model 

class VedioConverter():
    def __init__(self, number=5):
        self.time_F = number

    #一次把整部影片轉成圖片 並產生訓練用csv檔            
    def transform_to_traindata(self, filename, action):
        turn = 1
        train_images = []
        vedio = cv2.VideoCapture(filename)
        fps = vedio.get(cv2.CAP_PROP_FPS)
        tool = general_mulitpose_model()
        # f_csv = open("./trans_to_train/test.csv", "w", newline='') #測試用 到時候真正寫入檔案用追加方式(a+)
        f_csv = open("./trans_to_train/pose_data.csv", "a", newline='') #真正寫入檔案
        print("fps:",fps)

        while True:
            ret = vedio.grab()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            #read() = grab() + retrieve()
            #每隔幾幀進行擷取 省略每次都需read()的時間 改用grab + retrieve
            if(turn % self.time_F == 0):    #5
                ret, frame = vedio.retrieve() 
                if ret:
                    if(turn % (self.time_F*3) == 0):    #15 表示已經有3張圖了
                        train_images.append(frame)
                        dataset = tool.gen_train_dataset(train_images, action)                   
                        print(np.array(dataset).shape)
                        print(np.array(dataset))
                        if dataset: #有數據 表示是正常可用的 
                            tool.gen_train_csv(dataset, f_csv)  
                                
                        turn = 0
                        train_images.clear()
                        
                    else:
                        train_images.append(frame)
                else:
                    print("Can't receive frame (stream end?). Exiting ...") 
                    break    
            turn = turn + 1
        f_csv.close()
        vedio.release()   

    #把影片每3張圖 做一次預測資料 --預測用
    def transform_and_predict(self, filename):
        turn = 1
        predict_images = []
        vedio = cv2.VideoCapture(filename)
        

        while True:
            ret = vedio.grab()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # read() = grab() + retrieve()
            # 每隔幾幀進行擷取 省略每次都需read()的時間 改用grab + retrieve
            # 每n幀做擷取 每3n幀做擷取並把圖片集拿去預測一次 
            if(turn % self.time_F == 0):    #5
                ret, frame = vedio.retrieve() 
                if ret:
                    if(turn % (self.time_F*3) == 0):    #15 表示已經有3張圖了
                        predict_images.append(frame)
                        # *******未完成*******
                        # add predict code
                        # 圖片產生關鍵點信息 並判斷人數 人數跟之前一樣>繼續動作 不一樣>continue
                        # 將關鍵點信息 擷取特徵 並存入dataset
                        # ****dataset 拿去做預測***** 
                        # 銜接LSTM ......
                        # *******未完成*******
                        predict_images.clear()
                    else:
                        predict_images.append(frame)
                else:
                    print("Can't receive frame (stream end?). Exiting ...") 
                    break    
            turn = turn + 1

        vedio.release()   


if __name__ == '__main__':
    vc = VedioConverter()
    vc.transform_to_traindata("./train_images/Fishing_Training_Video/Produce.mp4", ["test"])

    cv2.destroyAllWindows()

