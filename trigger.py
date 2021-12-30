from Capturevediotoimage import VedioConverter 


curr_work_time = 0
curr_vedio_time = 0

if __name__ == '__main__':
    test = VedioConverter()
    test.test_predict("./train_images/v2/Produce_0.mp4", 15, 120)