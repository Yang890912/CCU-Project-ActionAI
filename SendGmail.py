from email.mime.text import MIMEText
import smtplib
import time

class SendGmail:
    def __init__(self, account, passwd,Recv):
        self.account = account
        self.passwd = passwd
        self.Recv = Recv
    def send_message(self):
        #gettime
        localtime = time.localtime()
        result = time.strftime("%Y-%m-%d %I:%M:%S %p", localtime)

        # MIME
        subject = "Warning"
        message = "[系統提醒]您的勞工已工作超時<br>\
                    <br>GMT+8 " + result 
        print(message)
        msg = MIMEText(message, "html")
        msg["Subject"] = subject
        msg["To"] = self.Recv
        msg["From"] = self.account
        
        # 寄信
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(self.account, self.passwd)
        server.send_message(msg)
        server.quit()
