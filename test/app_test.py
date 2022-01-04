import re
import unittest
import sys
sys.path.append('../')

from SendGmail import SendGmail
from trigger import GUI, EditWindow

class TestApp(unittest.TestCase):
    email1 = None
    email2 = None
    app = GUI()

    print()
    print("Please Enter two lines of email: test1@gmail.com and test2@hotmail.com for 'Edit Receiver Email' button sequentially")
    app.start()

    def test_gui_login_email_button(self):
        self.assertEqual(self.app.account.get(), "your_email_address")
        self.assertEqual(self.app.password.get(), "your_email_address_passwd")

    def test_gui_edit_email_button(self):
        with open("EmailList.txt", 'r') as file:
            email1 = file.readline().strip("\n")
            email2 = file.readline().strip("\n")
            file.close()

        self.assertEqual(email1, "test1@gmail.com")
        self.assertEqual(email2, "test2@hotmail.com")

    def test_gui_send_mail_button(self):
        self.assertEqual(self.app.err, None)

    def test_gui_run_button(self):
        self.assertEqual(self.app.is_over_work_thres, True)

if __name__ == '__main__':
    unittest.main(argv=['ignored', '-v'], exit=True)