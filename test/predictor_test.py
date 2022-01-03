import unittest
import sys
sys.path.append('../')
sys.path.append('./plot')

from Capturevideotoimage import VideoConverter
import predict_model_plot

def usage(script_name):
    print("python %s video_to_test_path rest_skiptime work_skiptime" % script_name)

if len(sys.argv) != 4:
    usage(sys.argv[0])
    exit()

class TestPrediction(unittest.TestCase):
    video_to_test = None
    rest_skiptime = None
    work_skiptime = None
    predictor = None

    def setUp(self):
        self.predictor = VideoConverter()
        self.video_to_test = sys.argv[1]
        self.rest_skiptime = int(sys.argv[2])
        self.work_skiptime = int(sys.argv[3])
        result_file = open("predict_result.txt", "a")
        result_file.truncate(0) # because maybe have previous video result

    def test_prediction(self):
        self.predictor.test_predict(self.video_to_test, self.rest_skiptime, self.work_skiptime)
        predict_model_plot.plot()

if __name__ == '__main__':
    unittest.main(argv=['ignored', '-v'], exit=True)