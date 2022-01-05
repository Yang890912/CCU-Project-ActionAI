
# CCU-project-ActionAI

- 此專案為中正大學軟體工程課程用，題目為**漁工過勞檢測系統**，透過使用者介面選擇資料夾，會歷遍資料夾影片並分析漁工工作狀態並計算時間，如有超時，會寄送電子郵件提醒使用者。

## Table of Contents
[TOC]

## Beginners Guide
- 下載位址最好為**英文路徑**，否則會出現無法預期的錯誤
- 環境為python 3.6.X
- python安裝時記得勾選，否則程式`trigger.py`會跑不起來  
![](https://i.imgur.com/7XOH7oO.png)

### 需要套件
- tensorflow==2.1.2
- h5py==2.10.0
- scipy
- scikit-learn
- **opencv-contrib-python**
- pandas
- schedule
- pillow
- matplotlib
### 安裝
    //安裝套件指令(從req檔案) 如有需要請另外安裝 如numpy...
    pip install -r requirements.txt	

**gpu環境**
- 此專案的**opencv**跟**tensorflow**套件預設使用CPU跑
如需要以GPU跑可參考CUDA跟cuDNN安裝指南跟opencv編譯指南

[CUDA](https://developer.nvidia.com/cuda-toolkit-archive)  
[cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)  
[CUDA&cuDNN安裝教學](https://medium.com/ching-i/win10-%E5%AE%89%E8%A3%9D-cuda-cudnn-%E6%95%99%E5%AD%B8-c617b3b76deb)  
[opencv編譯GPU版本教學](https://medium.com/chung-yi/build-opencv-gpu-version-on-windows-10-c37a33437525)  

## Pose Prediction(Developer)
- 我們用的是**OpenPose**開發中的**COCO model**來當作我們的姿勢預測模型，能夠分析出圖片的人並找出關鍵點  
[OpenPose github](https://github.com/CMU-Perceptual-Computing-Lab/openpose)  

**COCO model下載**  
[source](https://drive.google.com/file/d/1lBC7t3QKwu7Udp8t7kUda4K1zSbeW4JY/view?usp=sharing)  
下載完解壓縮後將`pose_deploy_linevec.prototxt`和`pose_iter_440000.caffemodel`丟到`model/coco/`資料夾即可  

**COCO model 執行結果**  
想執行 COCO model元件，可以執行`COCO_model.py`，會產生範例圖片跟預測結果，實際人數和期望值人數可以到`test/`資料夾執行 COCO_model_test.py（下面有更詳細解釋）  
```
$ python COCO_model.py
```
![](https://i.imgur.com/eMqel5R.png)  

## Train Data(Developer)  
- 有了上述的姿勢預測，我們能透過輸入影片，來擷取幀並預測，產生訓練資料，給之後的**action model**測試
- 在檔案`Capturevideotoimage.py`有三個函式可做使用，資料生成是用下面第1個函式
```
    # 輸入影片和動作，產生關鍵點資料到csv檔
    transform_to_traindata(self, filename, action)
    
    # 輸入影片，預測工時 
    transform_and_predict(self, filename)
    
    # 輸入影片，預測工時，可自訂工作跟休息要跳過的時間
    test_predict(self, filename, rest_skiptime, work_skiptime)
```
我們目前只有分**work**跟**rest**兩種動作，會將特徵寫入`trans_to_train/pose_data.csv`

## Action Prediction(Developer)
- 我們將動作預測模型寫在`action_model.py`，裡面實作各種函式包括訓練、預測、資料轉型...
```
gen_lstm_model(self)    # 產生lstm模型
load_data(self, filename)    # load檔案轉成訓練格式
train_lstm_model(self, model, X_train, y_train, X_test, y_test)    # 訓練並儲存
load_lstm_model(self, filename)    # load現成的模型
predict(self, dataset, lstm_model)    # 預測
```
直接執行`action_model.py`，會做訓練的動作

## User come here(User)
- 使用者可以透過執行`trigger.py`，產生GUI且選擇資料夾，程式會自動驅動各元件去做影片分析。
```
$ python trigger.py
```
執行後產生以下畫面  
![](https://i.imgur.com/qWhYQzO.png)


- **Open a Directory** : 可開啟要分想分析的影片的資料夾，程式會歷遍裡面的影片逐一分析。
- **Edit Receiver Email** : 設定**接收**過勞提醒的email帳號，會產生`EmailList.txt`，需填寫在裡面。
- **Send Email** : 手動按下可發送信件到`EmailList.txt`裡的所有信箱(測試發送功能用)。
- **Login Email** : 填寫要當**發送者**的email帳戶和密碼。
- **Run** : 按下後開始分析。
- **Reset** : 將目前累積的時間寫到`log.txt`
- 下面為顯示目前進度跟所在資料夾位置

### **Open a Directory**
產生類似下面介面，直接選取要的資料夾即可  
![](https://i.imgur.com/mIsgho5.png)

確保有選到正確的資料夾  
![](https://i.imgur.com/7iL6lIN.png)



### **Edit Receiver Email**
產生以下介面，輸入欲**接收信箱**即可(可多筆)。  
![image](https://user-images.githubusercontent.com/90774036/148155326-108660be-b2fa-46d0-a062-8c9501a481ac.png)

然後會在專案資料夾內產生txt檔紀錄  
![](https://i.imgur.com/ckjCdha.png)

### **Send Email(請先Login Email)**
會送出一封提醒郵件(手動測試信箱是否正確)，如下圖  
![](https://i.imgur.com/TA6CyEY.png)

### **Login Email**
產生以下介面，這邊輸入email帳密  
![](https://i.imgur.com/g6OImx5.png)

Email 格式有誤會警告  
![](https://i.imgur.com/J2aOTNA.png)


### **Run**
會顯示目前在分析的檔案，和分析累計時間(完成1個影片才會更新1次)  
![](https://i.imgur.com/WnRWn8Z.png)

命令列會輸出過程  
![](https://i.imgur.com/RqZIryr.png)

以上的`[Time]work/video`顯示的是進度的時間(範例影片只有10秒故可忽略結果)  

分析過的檔案，檔名會被標記`(done)`，不會重複分析  
![](https://i.imgur.com/aTDI2SP.png)

### **Reset**
目前程式有2種情況會記錄目前的累積時間到`log.txt`並把`Current Work Time`跟`Current Video Time`歸0，此按鈕為手動  
![](https://i.imgur.com/5cALq3k.png)  

![](https://i.imgur.com/2cCxnyi.png)

第2種情況是每天的23:59分會自動紀錄  

### **注意事項**
1. 我們採用的分析策略適合長時間影片(20分鐘以上)，如果是低於1分鐘的短片會效果不佳或是沒效果
2. 如果要在分析時切換資料夾，只能關掉程式重新選擇
3. 分析無法暫停，只能關掉並重新
4. `Current Work Time`跟`Current Video Time`在分析完單支影片才會更新，無法實時更新
5. 當`Current Work Time`大於8小時，就會自動發送信件。

## 測試
### COCO model
切換到`test/`資料夾，執行
```
$ python COCO_model_test.py expected_keypoint_cnt
```
這個測試會把 test/test_images 下 day 和 night 資料夾內的圖檔作姿勢預測分析
COCO model 會分析 18 種姿勢特徵，下圖為只要判斷到大於 "expected_keypoint_cnt" 姿勢特徵就算偵測到一個人，下圖為大於一半姿勢特徵（9種）在白天和晚上的準確率：
![](https://i.imgur.com/eFbgjz1.png)
**測試結果：白天比晚上的效果來得好**

### Action predictor model
切換到 `test/` 資料夾，執行
```
$ python predictor_test.py video_to_test_path rest_skiptime work_skiptime
```
video_to_test_path: 要分析的影片
rest_skiptime: 若偵測到休息，影片要前進的時間（單位為秒）
work_skiptime: 若偵測到工作，影片要前進的時間（單位為秒）

該模型測試了2個長影片（24分鐘和39分鐘，影片都是在白天拍攝），以下為分析影片是否在工作或休息的機率圖

- 24 分鐘影片（大部分時間都在工作）
    若偵測到休息，影片則前進30秒
    若偵測到工作，影片則前進
    - 1分鐘
    ![](https://i.imgur.com/VB0Hbh3.png)

    - 2分鐘
    ![](https://i.imgur.com/qAHadUo.png)

    - 5分鐘
    ![](https://i.imgur.com/oYRRG8s.png)

- 39 分鐘影片（大部分時間都在工作）
    若偵測到休息，影片則前進30秒
    若偵測到工作，影片則前進
    - 1分鐘
    ![](https://i.imgur.com/kPOUN1R.png)

    - 2分鐘
    ![](https://i.imgur.com/NxjeIYE.png)

    - 5分鐘
    ![](https://i.imgur.com/dfUcDdK.png)
    
**測試結果：若偵測到工作，影片則前進1分鐘對長影片的分析效果較佳**

### 使用者介面
![](https://i.imgur.com/wVJd2Np.png)

換到 `test/` 資料夾，請先打開 app_test.py 檔案找到 `test_gui_login_email_button` 函數並把裡面的 `your_email_address` 和 `your_email_address_password` 改成你合法的信箱和信箱密碼，執行
```
$ python app_test.py
```
Login Email 按鈕：按下 Login Email 按鈕輸入剛剛修改的合法信箱和密碼

Edit Receiver Email 按鈕：按下 Edit Receiver Email 按鈕輸入接收者的信箱（一行一個信箱），第一第二個信箱請輸入終端的提示訊息的信箱（為了測試），其他行數可以是其他信箱

Send Email 按鈕：按下 Send Email 按鈕會發送超時訊息到 Edit Receiver Email 設定的接收者

Run 按鈕：按下後會開始分析選擇的資料夾內所有未分析過的影片，若工作累計時數超過預設工作時數上限（8小時）就會發送工作超時訊息給 EmailList.txt 內設定的信箱

**測試結果：OK**


