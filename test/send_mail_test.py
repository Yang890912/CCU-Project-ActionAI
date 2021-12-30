import unittest
import sys
sys.path.append('../')

from SendGmail import SendGmail

def usage(script_name):
    print("python %s threshold email_recv" % script_name)

if len(sys.argv) != 3:
    usage(sys.argv[0])
    exit()

class TestSendMail(unittest.TestCase):
    thres = None
    recver = None
    snder = None

    def setUp(self):
        self.thres = int(sys.argv[1])
        self.recver = sys.argv[2]
        self.snder = SendGmail("test88812345@gmail.com", 88812345, self.recver)

    # 當 model 預測工作超時的時候，發送 email
    def test_send_mail_with_thres(self):
        print("todo")

if __name__ == '__main__':
    unittest.main(argv=['ignored', '-v'], exit=True)