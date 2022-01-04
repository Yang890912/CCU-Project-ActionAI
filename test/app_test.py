import re
import unittest
import sys
sys.path.append('../')

from SendGmail import SendGmail
from trigger import GUI, EditWindow

# def usage(script_name):
#     print("python %s video_to_test_path rest_skiptime work_skiptime threshold email_recv" % script_name)

def email_validate(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.fullmatch(regex, email)

# if len(sys.argv) != 6:
#     usage(sys.argv[0])
#     exit()

# if not email_validate(sys.argv[5]):
#     print("Invalid receiver email address")
#     exit()

class TestSendMail(unittest.TestCase):
    app = GUI()

    print()
    print("Please Enter email: abc@gmail.com, password: 123456 for 'Login Email button'")
    print("Please Enter two lines of email: test1@gmail.com and test2@hotmail.com for 'Edit Receiver Email' sequentially")
    app.start()

    def test_app_button(self):
        self.assertEqual(self.app.account.get(), "abc@gmail.com")
        self.assertEqual(self.app.password.get(), "123456")

        email1 = None
        email2 = None

    def test_gui_edit_email(self):
        with open("EmailList.txt", 'r') as file:
            email1 = file.readline().strip("\n")
            email2 = file.readline().strip("\n")

        self.assertEqual(email1, "test1@gmail.com")
        self.assertEqual(email2, "test2@hotmail.com")


if __name__ == '__main__':
    unittest.main(argv=['ignored', '-v'], exit=True)