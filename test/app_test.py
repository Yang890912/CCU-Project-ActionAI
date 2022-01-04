import re
import unittest
import sys
sys.path.append('../')

from SendGmail import SendGmail
from trigger import GUI, EditWindow

# def usage(script_name):
#     print("python %s your_email_address your_emaill_address_password" % script_name)

# def email_validate(email):
#     regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
#     return re.fullmatch(regex, email)

# if len(sys.argv) != 3:
#     usage(sys.argv[0])
#     exit()

# # if not email_validate(sys.argv[5]):
#     print("Invalid receiver email address")
#     exit()

class TestSendMail(unittest.TestCase):
    email1 = None
    email2 = None
    app = GUI()

    print()
    print("Please Enter two lines of email: test1@gmail.com and test2@hotmail.com for 'Edit Receiver Email' sequentially")
    app.start()

    def test_gui_login_email_button(self):
        self.assertEqual(self.app.account.get(), "test88812345@gmail.com")
        self.assertEqual(self.app.password.get(), "1234567a.")
        # self.assertEqual(self.app.account.get(), "your_email_address")
        # self.assertEqual(self.app.password.get(), "your_email_address_passwd")

    def test_gui_edit_email_button(self):
        with open("EmailList.txt", 'r') as file:
            email1 = file.readline().strip("\n")
            email2 = file.readline().strip("\n")
            file.close()

        self.assertEqual(email1, "test1@gmail.com")
        self.assertEqual(email2, "test2@hotmail.com")

    def test_gui_send_mail_button(self):
        email_list = open("EmailList.txt")
        Receivers = email_list.readlines()
        for Recv in Receivers:
            SG = SendGmail(self.app.Account, self.app.Password, Recv)
            Failed = SG.send_message()
            self.assertFalse(Failed)
        email_list.close()

if __name__ == '__main__':
    unittest.main(argv=['ignored', '-v'], exit=True)