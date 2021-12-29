import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense , LSTM , Dropout , Bidirectional 
from tensorflow.keras import Sequential 
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop

class PosePredictor():
    def __init__(self):
        self.window = 3
        self.pose_vec_dim = 20
        self.list_of_action = ['work','rest']  # 需調整
        self.lbl_dict = {class_name:idx for idx, class_name in enumerate(self.list_of_action)}

    def load_lstm_model(self, filename):
        model = load_model(filename)
        return model

    def gen_lstm_model(self):   # 參數可調整 暫定
        model = Sequential()
        model.add(LSTM(units=32, return_sequences=True , input_shape=(self.window, self.pose_vec_dim)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=32, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=32))
        model.add(Dropout(0.2))
        model.add(Dense(units=len(self.list_of_action), activation='softmax')) # 輸出層

        print(model.summary())
        return model

    def train_lstm_model(self, model, X_train, y_train, X_test, y_test):   #參數可調整 暫定
        model.compile(loss='categorical_crossentropy', 
                      optimizer=RMSprop(), 
                      metrics=['accuracy'])

        history = model.fit(X_train , y_train,
                            epochs=200,   # 迭代次數
                            batch_size=64, 
                            verbose=1,    # 1 輸出進度條
                            validation_data=(X_test, y_test)) # 驗證資料 
        
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model.save("./model/lstm_fishman_action.h5")
        print("Saved model to disk")

    def load_data(self, filename):
        dataset = pd.read_csv(filename, thousands=",", encoding='utf-8')
        y = dataset.pop('action')   
        dataset = dataset.drop(['person','person.1','person.2'] , axis=1)   # 刪掉不要的column
        X = dataset.values  # 讀完後格式為np.array

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        y_train = to_categorical(list(map(self.lbl_dict.get, y_train)), len(self.list_of_action))   # 轉完後格式為np.array
        y_test = to_categorical(list(map(self.lbl_dict.get, y_test)), len(self.list_of_action))

        X_test = X_test.reshape(X_test.shape[0], self.window, self.pose_vec_dim)    # 2維轉3維
        X_train = X_train.reshape(X_train.shape[0], self.window, self.pose_vec_dim)
        return X_train, X_test, y_train, y_test

    def predict(self, dataset, lstm_model):
        dataset = dataset.reshape(dataset.shape[0], self.window, self.pose_vec_dim)
        result = lstm_model.predict(dataset)
        return result
    

if __name__ == '__main__':
    predictor = PosePredictor()
    model = predictor.gen_lstm_model()
    X_train, X_test, y_train, y_test = predictor.load_data("./trans_to_train/pose_data.csv")
    print(X_train)
    print(y_train)

    predictor.train_lstm_model(model, X_train, y_train, X_test, y_test)