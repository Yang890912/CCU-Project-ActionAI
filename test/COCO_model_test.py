import unittest
import os
import glob
import sys
sys.path.append('../')
sys.path.append('./plot')

from COCO_model import general_mulitpose_model
import COCO_model_plot

def usage(script_name):
    print("python %s expected_keypoint_cnt" % script_name)

if len(sys.argv) != 2:
    usage(sys.argv[0])
    exit()

# day images path
test_day_imgs = glob.glob(os.getcwd() + "/test_images/day/**/*.png", recursive=True)

# day images cnt
test_day_imgs_cnt_dict = {
    'front': len([img for img in os.listdir('./test_images/day/front/') if img.find(".png") != -1]),
    'rear': len([img for img in os.listdir('./test_images/day/rear/') if img.find(".png") != -1]),
    'side': len([img for img in os.listdir('./test_images/day/side/') if img.find(".png") != -1]) 
}

# day images expected people cnt
test_day_imgs_expected = []
with open('./test_images/day/expected.txt') as file:
    for line in file: 
        test_day_imgs_expected.append(int(line.strip()))
    file.close()
    
# night images path
test_night_imgs = glob.glob(os.getcwd() + "/test_images/night/**/*.png", recursive=True)

# night images cnt
test_night_imgs_cnt_dict = {
    'front': len([img for img in os.listdir('./test_images/night/front/') if img.find(".png") != -1]),
    'rear': len([img for img in os.listdir('./test_images/night/rear/') if img.find(".png") != -1]),
    'side': len([img for img in os.listdir('./test_images/night/side/') if img.find(".png") != -1]) 
}

# night images expected people cnt
test_night_imgs_expected = []
with open('./test_images/night/expected.txt') as file:
    for line in file: 
        test_night_imgs_expected.append(int(line.strip()))
    file.close()

class TestCOCOModel(unittest.TestCase):
    def setUp(self):
        self.verifications_err = []
        self.test_day_front_avg = 0
        self.test_day_side_avg = 0
        self.test_day_rear_avg = 0
        self.test_night_front_avg = 0
        self.test_night_side_avg = 0
        self.test_night_rear_avg = 0

    def tearDown(self):
        self.assertEqual([], self.verifications_err)

    def test_COCO_Model_Day(self):
        self.assertEqual.__self__.maxDiff = None

        multipose_model = general_mulitpose_model()
        front_msg = "day front's accuracy of "
        rear_msg = "day rear's accuracy of "
        side_msg = "day side's accuracy of "
        
        idx = 0
        for img in test_day_imgs:
            basename = os.path.basename(img)
            multipose_model.predict(img)
            detected_people_cnt = multipose_model.getPeopleCntByKeyPointsFile(expected_keypoint_cnt=int(sys.argv[1]))
            expected_people_cnt = test_day_imgs_expected[idx]
            accuracy = detected_people_cnt / expected_people_cnt * 100
            if accuracy > 100.0:
                accuracy = expected_people_cnt / detected_people_cnt * 100
            idx += 1

            if "front" in img:
                self.test_day_front_avg += accuracy
                try: 
                    self.assertEqual(detected_people_cnt, expected_people_cnt, msg=front_msg + "(" + basename + "): " + str(accuracy) + "%")
                except AssertionError as e: 
                    self.verifications_err.append(str(e))
            elif "rear" in img:
                self.test_day_rear_avg += accuracy
                try: 
                    self.assertEqual(detected_people_cnt, expected_people_cnt, msg=rear_msg + "(" + basename + "): " + str(accuracy) + "%")
                except AssertionError as e: 
                    self.verifications_err.append(str(e))
            elif "side" in img:
                self.test_day_side_avg += accuracy
                try: 
                    self.assertEqual(detected_people_cnt, expected_people_cnt, msg=side_msg + "(" + basename + "): " + str(accuracy) + "%")
                except AssertionError as e: 
                    self.verifications_err.append(str(e))
                
        self.test_day_front_avg /= test_day_imgs_cnt_dict.get('front')
        self.test_day_rear_avg /= test_day_imgs_cnt_dict.get('rear')
        self.test_day_side_avg /= test_day_imgs_cnt_dict.get('side')

        print("day front's average accuracy: {}%".format(self.test_day_front_avg))
        print("day rear's average accuracy: {}%".format(self.test_day_rear_avg))
        print("day side's average accuracy: {}%".format(self.test_day_side_avg))

        with open("./test_images/day/res.txt", "w") as res:
            res.write(str(self.test_day_front_avg))
            res.write("\n")
            res.write(str(self.test_day_rear_avg))
            res.write("\n")
            res.write(str(self.test_day_side_avg))
            res.write("\n")
            res.close()

    def test_COCO_Model_Night(self):
        self.assertEqual.__self__.maxDiff = None

        multipose_model = general_mulitpose_model()
        front_msg = "night front's accuracy of "
        rear_msg = "night rear's accuracy of "
        side_msg = "night side's accuracy of "
        
        idx = 0
        for img in test_night_imgs:
            basename = os.path.basename(img)
            multipose_model.predict(img)
            detected_people_cnt = multipose_model.getPeopleCntByKeyPointsFile(expected_keypoint_cnt=int(sys.argv[1]))
            expected_people_cnt = test_night_imgs_expected[idx]
            accuracy = detected_people_cnt / expected_people_cnt * 100
            if accuracy > 100.0:
                accuracy = expected_people_cnt / detected_people_cnt * 100
            idx += 1

            if "front" in img:
                self.test_night_front_avg += accuracy
                try: 
                    self.assertEqual(detected_people_cnt, expected_people_cnt, msg=front_msg + "(" + basename + "): " + str(accuracy) + "%")
                except AssertionError as e: 
                    self.verifications_err.append(str(e))
            elif "rear" in img:
                self.test_night_rear_avg += accuracy
                try: 
                    self.assertEqual(detected_people_cnt, expected_people_cnt, msg=rear_msg + "(" + basename + "): " + str(accuracy) + "%")
                except AssertionError as e: 
                    self.verifications_err.append(str(e))
            elif "side" in img:
                self.test_night_side_avg += accuracy
                try: 
                    self.assertEqual(detected_people_cnt, expected_people_cnt, msg=side_msg + "(" + basename + "): " + str(accuracy) + "%")
                except AssertionError as e: 
                    self.verifications_err.append(str(e))
                
        self.test_night_front_avg /= test_night_imgs_cnt_dict.get('front')
        self.test_night_rear_avg /= test_night_imgs_cnt_dict.get('rear')
        self.test_night_side_avg /= test_night_imgs_cnt_dict.get('side')

        print("night front's average accuracy: {}%".format(self.test_night_front_avg))
        print("night rear's average accuracy: {}%".format(self.test_night_rear_avg))
        print("night side's average accuracy: {}%".format(self.test_night_side_avg))

        with open("./test_images/night/res.txt", "w") as res:
            res.write(str(self.test_night_front_avg))
            res.write("\n")
            res.write(str(self.test_night_rear_avg))
            res.write("\n")
            res.write(str(self.test_night_side_avg))
            res.write("\n")
            res.close()

    def test_plot(self):
        COCO_model_plot.plot()

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=True)