#Refer:
#[1] https://www.aiuai.cn/aifarm946.html

import os
from os.path import dirname, join
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


class general_mulitpose_model(object):
    def __init__(self):
        self.point_names = ['Nose', 'Neck',
                            'R-Sho', 'R-Elb', 'R-Wr',
                            'L-Sho', 'L-Elb', 'L-Wr',
                            'R-Hip', 'R-Knee', 'R-Ank',
                            'L-Hip', 'L-Knee', 'L-Ank',
                            'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
        self.point_pairs = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
                            [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
                            [1,0], [0,14], [14,16], [0,15], [15,17],
                            [2,17], [5,16] ]
        #2 右手臂 3 右手腕 4 左手臂 5 左手腕 
        #7 右大腿 8 右小腿 10 左大腿 11 左小腿 

        # index of pafs correspoding to the self.point_pairs
        # e.g for point_pairs(1,2), the PAFs are located at indices (31,32) of output,
        #   Similarly, (1,5) -> (39,40) and so on.
        self.map_idx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
                        [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
                        [47,48], [49,50], [53,54], [51,52], [55,56],
                        [37,38], [45,46]]

        self.colors = [[0,100,255], [0,100,255],   [0,255,255],
                       [0,100,255], [0,255,255],   [0,100,255],
                       [0,255,0],   [255,200,100], [255,0,255],
                       [0,255,0],   [255,200,100], [255,0,255],
                       [0,0,255],   [255,0,0],     [200,200,0],
                       [255,0,0],   [200,200,0],   [0,0,0]]

        self.num_points = 18
        self.pose_net = self.get_model()    #模組提取


    #模組提取
    def get_model(self):
        prototxt = join(dirname(__file__), "./model/coco/pose_deploy_linevec.prototxt")
        caffemodel = join(dirname(__file__), "./model/coco/pose_iter_440000.caffemodel")
        coco_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        coco_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        coco_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return coco_net


    def getKeypoints(self, probMap, threshold=0.1):
        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
        mapMask = np.uint8(mapSmooth > threshold)

        keypoints = []
        # find the blobs
        contours, hierarchy = cv2.findContours(mapMask,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)

        # for each blob find the maxima
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

        return keypoints


    def getValidPairs(self, output, detected_keypoints, img_width, img_height):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7

        for k in range(len(self.map_idx)):
            # A->B constitute a limb
            pafA = output[0, self.map_idx[k][0], :, :]
            pafB = output[0, self.map_idx[k][1], :, :]
            pafA = cv2.resize(pafA, (img_width, img_height))
            pafB = cv2.resize(pafB, (img_width, img_height))

            # Find the keypoints for the first and second limb
            candA = detected_keypoints[self.point_pairs[k][0]]
            candB = detected_keypoints[self.point_pairs[k][1]]
            nA = len(candA)
            nB = len(candB)

            if (nA != 0 and nB != 0):
                valid_pair = np.zeros((0, 3))
                for i in range(nA):
                    max_j = -1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        # Find d_ij
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(
                            zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(
                                round(interp_coord[k][0]))],
                                               pafB[int(round(interp_coord[k][1])), int(
                                                   round(interp_coord[k][0]))]])
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores) / len(paf_scores)

                        # Check if the connection is valid
                        if (len(np.where(paf_scores > paf_score_th)[
                                    0]) / n_interp_samples) > conf_th:
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1

                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(valid_pair,
                                               [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)
            else:  # If no keypoints are detected
                #print("No Connection : k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])

        return valid_pairs, invalid_pairs


    def getPersonwiseKeypoints(self, valid_pairs, invalid_pairs, keypoints_list):
        personwiseKeypoints = -1 * np.ones((0, 19))

        for k in range(len(self.map_idx)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:, 0]
                partBs = valid_pairs[k][:, 1]
                indexA, indexB = np.array(self.point_pairs[k])

                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += keypoints_list[
                                                                   partBs[i].astype(int), 2] + \
                                                               valid_pairs[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score
                        row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + \
                                  valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])

        return personwiseKeypoints

    # 將偵測到的 keypoints 寫到檔案，用於測試
    def writeKeyPointsToFile(self, keypoints, filename="keypoints.txt"):
        with open(filename, 'a') as file:
            file.truncate(0)
            for keypoint in keypoints:
                for kp in keypoint:
                    file.write(''.join(str(kp)))
                    file.write(',')
                file.write('\n')
            file.close()

    # 根據 keypoints 檔案，計算人數
    def getPeopleCntByKeyPointsFile(self, expected_keypoint_cnt, filename="keypoints.txt"):
        people_cnt = 0
        keypoints_cnt = []
        with open(filename, 'r') as file:
            for line in file:
                tmp = []
                start = line.find("),")
                while start != -1:
                    tmp.append(1)
                    start += 1
                    start = line.find("),", start)

                keypoints_cnt.append(tmp)

        # find the largest size of row in keypoints_cnt
        largest_row_cnt = 0
        for row in keypoints_cnt:
            if len(row) > largest_row_cnt:
                largest_row_cnt = len(row)

        # insert 0 to others to fit the array
        for row in keypoints_cnt:
            while len(row) != largest_row_cnt:
                row.append(0)

        # sum up the col, if sum larger than 9(half of the characteristics), 
        # then someone is detected
        for col in zip(*keypoints_cnt):
            cnt = np.sum(col)
            if cnt >= expected_keypoint_cnt:
                people_cnt += 1

        return people_cnt

    # 將圖片經過 COCO model 做預測 --字串格式
    def predict(self, inputparam):
        img_cv2 = cv2.imread(inputparam)
        img_width, img_height = img_cv2.shape[1], img_cv2.shape[0]

        #調整圖片大小 長368
        net_height = 368
        net_width = int((net_height / img_height) * img_width)

        start = time.time()
        in_blob = cv2.dnn.blobFromImage(
            img_cv2,
            1.0 / 255,
            (net_width, net_height),
            (0, 0, 0),
            swapRB=False,
            crop=False)

        #input 跟 output 格式
        self.pose_net.setInput(in_blob)
        output = self.pose_net.forward()
        print("[INFO]Time Taken in Forward pass: {}".format(time.time() - start))

        output_keypoints = []
        detected_keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0
        threshold = 0.1
        for part in range(self.num_points):
            probMap = output[0, part, :, :]
            probMap = cv2.resize(probMap, (img_cv2.shape[1], img_cv2.shape[0]))
            keypoints = self.getKeypoints(probMap, threshold)
            # print("Keypoints - {} : {}".format(self.point_names[part], keypoints))
            output_keypoints.append(keypoints)

            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = \
            self.getValidPairs(output, 
                               detected_keypoints, 
                               img_width, 
                               img_height)
        personwiseKeypoints = \
            self.getPersonwiseKeypoints(valid_pairs, 
                                        invalid_pairs, 
                                        keypoints_list)

        self.writeKeyPointsToFile(output_keypoints)

        return personwiseKeypoints, keypoints_list
    
    # 將圖片經過 COCO model 做預測 --檔案格式
    def predict_v2(self, img_cv2):
        img_width, img_height = img_cv2.shape[1], img_cv2.shape[0]

        #調整圖片大小 長368
        net_height = 368
        net_width = int((net_height / img_height) * img_width)

        start = time.time()
        in_blob = cv2.dnn.blobFromImage(
            img_cv2,
            1.0 / 255,
            (net_width, net_height),
            (0, 0, 0),
            swapRB=False,
            crop=False)

        #input 跟 output 格式
        self.pose_net.setInput(in_blob)
        output = self.pose_net.forward()
        #print("[INFO]Time Taken in Forward pass: {}".format(time.time() - start))

        detected_keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0
        threshold = 0.1
        for part in range(self.num_points):
            probMap = output[0, part, :, :]
            probMap = cv2.resize(probMap, (img_cv2.shape[1], img_cv2.shape[0]))
            keypoints = self.getKeypoints(probMap, threshold)
            #print("Keypoints - {} : {}".format(self.point_names[part], keypoints))
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = \
            self.getValidPairs(output, 
                               detected_keypoints, 
                               img_width, 
                               img_height)
        personwiseKeypoints = \
            self.getPersonwiseKeypoints(valid_pairs, 
                                        invalid_pairs, 
                                        keypoints_list)

        self.writeKeyPointsToFile(output_keypoints)

        return personwiseKeypoints, keypoints_list

    # 可視化 --test
    def vis_pose(self, img_file, personwiseKeypoints, keypoints_list):
        img_cv2 = cv2.imread(img_file)
        for n in range(len(personwiseKeypoints)):
            for i in range(17):
                index = personwiseKeypoints[n][np.array(self.point_pairs[i])]   #看看是否有這個pair
                if -1 in index: #一個點沒有就不要畫線
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0]) 
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(img_cv2, (B[0], A[0]), (B[1], A[1]), self.colors[i], 3, cv2.LINE_AA)

        plt.figure()
        plt.imshow(img_cv2[:, :, ::-1])
        plt.title("Results")
        plt.axis("off")
        plt.show()
    
    # 將關鍵節點轉換成特徵向量
    def gen_pose(self, personwiseKeypoints, keypoints_list):
        rows = []
        throw = [0,1,12,13,14,15,16]
        for n in range(len(personwiseKeypoints)):   #第n個人
            #多新的的一列
            #人的編號當一個column
            rows.append([])     
            rows[n].append(n)   

            for i in range(17):
                if i in throw: 
                    continue

                #看看是否有這個pair
                index = personwiseKeypoints[n][np.array(self.point_pairs[i])]  

                #一個點沒有就不要連接
                if -1 in index: 
                    rows[n].append(-1)
                    rows[n].append(-1)
                    continue
                    
                B = np.int32(keypoints_list[index.astype(int), 0]) 
                A = np.int32(keypoints_list[index.astype(int), 1])
                rows[n].append(B[1]-B[0])
                rows[n].append(A[1]-A[0])
        return rows

    # 產生訓練資料csv檔 --test
    def gen_pose_csv(self, personwiseKeypoints, keypoints_list):
        header = ["person", 
                  "x-右手臂", "y-右手臂", "x-右手腕", "y-右手腕", 
                  "x-左手臂", "y-左手臂", "x-左手腕", "y-左手腕", 
                  "x-右軀幹", "y-右軀幹", "x-右大腿", "y-右大腿", "x-右小腿", "y-右小腿", 
                  "x-左軀幹", "y-左軀幹", "x-左大腿", "y-左大腿", "x-左小腿", "y-左小腿"]
        # header = ["person", "0", "1", 
        #           "x-右手臂", "y-右手臂", "x-右手腕", "y-右手腕", 
        #           "x-左手臂", "y-左手臂", "x-左手腕", "y-左手腕", 
        #           "x-右軀幹", "y-右軀幹", "x-右大腿", "y-右大腿", "x-右小腿", "y-右小腿", 
        #           "x-左軀幹", "y-左軀幹", "x-左大腿", "y-左大腿", "x-左小腿", "y-左小腿", "12", "13", "14", "15", "16"]
        rows = []
        throw = [0,1,12,13,14,15,16]
        for n in range(len(personwiseKeypoints)):   #第n個人
            #多新的的一列
            #人的編號當一個column
            rows.append([])     
            rows[n].append(n)   

            for i in range(17):
                if i in throw: 
                    continue

                #看看是否有這個pair
                index = personwiseKeypoints[n][np.array(self.point_pairs[i])]  

                #一個點沒有就不要連接
                if -1 in index: 
                    rows[n].append(-1)
                    rows[n].append(-1)
                    continue
                    
                B = np.int32(keypoints_list[index.astype(int), 0]) 
                A = np.int32(keypoints_list[index.astype(int), 1])
                rows[n].append(B[1]-B[0])
                rows[n].append(A[1]-A[0])
                #cv2.line(img_cv2, (B[0], A[0]), (B[1], A[1]), self.colors[i], 3, cv2.LINE_AA)
        print(rows)
        f = open("./trans_to_train/test.csv", "w", newline='')
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
        f.close()

    # 產生訓練資料csv檔  
    def gen_train_csv(self, dataset, file):
        # csv檔header排列... 表示同1個人3個時間軸的身體特徵
        # ["person", 
        # "x-右手臂", "y-右手臂", "x-右手腕", "y-右手腕", 
        # "x-左手臂", "y-左手臂", "x-左手腕", "y-左手腕", 
        # "x-右軀幹", "y-右軀幹", "x-右大腿", "y-右大腿", "x-右小腿", "y-右小腿", 
        # "x-左軀幹", "y-左軀幹", "x-左大腿", "y-左大腿", "x-左小腿", "y-左小腿",
        # "person", 
        # "x-右手臂", "y-右手臂", "x-右手腕", "y-右手腕", 
        # "x-左手臂", "y-左手臂", "x-左手腕", "y-左手腕", 
        # "x-右軀幹", "y-右軀幹", "x-右大腿", "y-右大腿", "x-右小腿", "y-右小腿", 
        # "x-左軀幹", "y-左軀幹", "x-左大腿", "y-左大腿", "x-左小腿", "y-左小腿",
        # "person", 
        # "x-右手臂", "y-右手臂", "x-右手腕", "y-右手腕", 
        # "x-左手臂", "y-左手臂", "x-左手腕", "y-左手腕", 
        # "x-右軀幹", "y-右軀幹", "x-右大腿", "y-右大腿", "x-右小腿", "y-右小腿", 
        # "x-左軀幹", "y-左軀幹", "x-左大腿", "y-左大腿", "x-左小腿", "y-左小腿",
        # "action"]
        w = csv.writer(file)
        w.writerows(dataset)
   
    # 將圖片集轉成特徵資料 並接上LSTM (然後將預測結果輸出) 
    def gen_train_dataset(self, imageset, action):
        image_num = np.array(imageset).shape[0]
        person_num = 0  #紀錄這些圖片裡應有幾人 判斷圖片人數都一樣才做計算
        min_person = 0 #小於此人數直接不記
        dataset = []

        # 依序讀取圖 並轉成關鍵點資訊 2維資料 人*(特徵*時間) 預設時間片段為3
        # 3張圖代表3個時間軸 人各自的特徵關係
        for i in range(0, image_num):
            personwiseKeypoints, keypoints_list = self.predict_v2(imageset[i])
            self.vis_pose(imageset[i], personwiseKeypoints, keypoints_list)
            curr_person = personwiseKeypoints.shape[0]
            if(i == 0):
                if(curr_person <= min_person): # 第1張圖小於最少人數 就直接不算了
                    return
                else:
                    person_num = curr_person
            elif(person_num != curr_person):   # 後2張圖人數跟第1張圖不一樣 直接不算 (暫不考慮人數變化的影響，直接忽略一次計算)
                return

            data = self.gen_pose(personwiseKeypoints, keypoints_list)
            if(i == 0):
                dataset = data
            else:
                for j in range(0, person_num):
                    dataset[j].extend(data[j])
                    if(i==image_num-1):
                        dataset[j].extend(action)

        # 接著把dataset丟入LSTM做預測 
        # dataset為2維資料 row是每個人 column是特徵的3個時間軸 
        # 因此要轉成3維資料 這邊可以在LSTM的程式碼那邊去做轉換 
        # 轉成 人*時間軸*特徵 可用 reshape實作 
        
        return dataset

    def gen_predict_dataset(self, imageset):
        image_num = np.array(imageset).shape[0]
        person_num = 0  #紀錄這些圖片裡應有幾人 判斷圖片人數都一樣才做計算
        min_person = 0 #小於此人數直接不記
        dataset = []

        # 依序讀取圖 並轉成關鍵點資訊 2維資料 人*(特徵*時間) 預設時間片段為3
        # 3張圖代表3個時間軸 人各自的特徵關係
        for i in range(0, image_num):
            personwiseKeypoints, keypoints_list = self.predict_v2(imageset[i])
            # self.vis_pose(imageset[i], personwiseKeypoints, keypoints_list)
            curr_person = personwiseKeypoints.shape[0]
            if(i == 0):
                if(curr_person <= min_person): # 第1張圖小於最少人數 就直接不算了
                    return
                else:
                    person_num = curr_person
            elif(person_num != curr_person):   # 後2張圖人數跟第1張圖不一樣 直接不算 (暫不考慮人數變化的影響，直接忽略一次計算)
                return

            data = self.gen_pose(personwiseKeypoints, keypoints_list)
            data.pop(0)
            if(i == 0): # 第一張圖 直接放入 
                dataset = data
            else:
                for j in range(0, person_num):
                    dataset[j].extend(data[j])

        dataset = np.array(dataset) # numpy type

        # 接著把dataset丟入LSTM做預測 
        # dataset為2維資料 row是每個人 column是特徵的3個時間軸 
        # 因此要轉成3維資料 這邊可以在LSTM的程式碼那邊去做轉換 
        # 轉成 人*時間軸*特徵 可用 reshape實作 
        
        return dataset 

if __name__ == '__main__':
    print("[INFO]MultiPose estimation.")
    # img_file = "D:/pythonTool/openpose/examples/media/COCO_val2014_000000000192.jpg"
    img_file = "./test/test_images/day/front/3.png"
    
    start = time.time()
    multipose_model = general_mulitpose_model()
    print("[INFO]Time Taken in Model Loading: {}".format(time.time() - start))
    personwiseKeypoints, keypoints_list = multipose_model.predict(img_file)
    # 對照圖
    multipose_model.vis_pose(img_file, 
                             personwiseKeypoints, 
                             keypoints_list)
    # 數據轉換成csv --test
    multipose_model.gen_pose_csv(personwiseKeypoints, 
                                 keypoints_list)

    #print(personwiseKeypoints)
    #print(keypoints_list.shape)
    print("[INFO]Time Taken in Done: {}".format(time.time() - start))
    print("[INFO]Done.")