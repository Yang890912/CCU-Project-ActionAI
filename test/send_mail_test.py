import re
import unittest
import sys
sys.path.append('../')

from SendGmail import SendGmail
from Capturevideotoimage import VideoConverter

def usage(script_name):
    print("python %s video_to_test_path rest_skiptime work_skiptime threshold email_recv" % script_name)

def email_validate(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.fullmatch(regex, email)

if len(sys.argv) != 6:
    usage(sys.argv[0])
    exit()

if not email_validate(sys.argv[5]):
    print("Invalid receiver email address")
    exit()

class TestSendMail(unittest.TestCase):
    video_to_test = None
    rest_skiptime = None
    work_skiptime = None
    thres = None
    recver = None
    snder = None
    predictor = None

    def setUp(self):
        self.predictor = VideoConverter()
        self.video_to_test = sys.argv[1]
        self.rest_skiptime = int(sys.argv[2])
        self.work_skiptime = int(sys.argv[3])
        self.thres = int(sys.argv[4])
        self.recver = sys.argv[5]
        self.snder = SendGmail("your_email", "your_email_password", self.recver)

    # 當 model 預測工作超時的時候，發送 email
    def test_send_mail_with_thres(self):
        worktime, videotime = self.predictor.test_predict(self.video_to_test, self.rest_skiptime, self.work_skiptime)
        try: 
            self.assertGreaterEqual(worktime, self.thres)
            is_failed = self.snder.send_message()
            self.assertFalse(is_failed)
        except AssertionError as e: 
            print("No reach the threshold")

if __name__ == '__main__':
    unittest.main(argv=['ignored', '-v'], exit=True)