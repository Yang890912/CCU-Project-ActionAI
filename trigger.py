import os
import re
import time
import threading
import signal
import schedule
import datetime
import tkinter as tk
from Capturevideotoimage import VideoConverter
from SendGmail import SendGmail
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

# /mnt/d/my_code/CCU-project-ActionAI

def email_validate(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.fullmatch(regex, email)

class GUI():
    def __init__(self):
        print('Initialize ...')
        self.root = tk.Tk()
        self.root.title('Overwork Detection')
        self.root.resizable(False, False)
        self.root.geometry('500x350')
        self.DirPath = os.path.abspath(os.getcwd())
        self.Account = str()
        self.Password = str()
        self.Running = False
        self.CurrentWorkTime = 0
        self.CurrentVideoTime = 0
        self.WorkTimeStr = StringVar()
        self.VideoTimeStr = StringVar()
        self.CurrentDir = StringVar()
        self.CurrentVideo = StringVar()
        self.CurrentStatus = StringVar()
        self.WorkTimeStr.set('Current Work Time = ' + self.SecondToStr(self.CurrentWorkTime))
        self.VideoTimeStr.set('Current Video Time = ' + self.SecondToStr(self.CurrentVideoTime))
        self.CurrentDir.set('Current Directory: ' + self.DirPath)
        self.CurrentVideo.set('Current Predict Video: ')
        self.CurrentStatus.set('Current Status: Not Working')
        self.err = None
        self.work_thres = 28800 # 8 hours
        self.is_over_work_thres = False
        self.stop_event = threading.Event()
        self.pid = os.getpid()
        
        self.PredictThread = threading.Thread(target = self.search_new_file)
        self.ResetEveryDayThread = threading.Thread(target = self.reset_everyday)
        # schedule.every().minute.at(":30").do(self.reset) # for test
        schedule.every().day.at("23:59").do(self.reset)

        # overwrite X callback
        self.root.protocol('WM_DELETE_WINDOW', self.terminate)

        self.open_button = ttk.Button(
            self.root,
            text = 'Open a Directory',
            width = 20,
            command = self.select_dir
        )
        self.run_button = ttk.Button(
            self.root,
            text = 'Run',
            width = 20,
            command = self.run_predict_thread
        )
        self.edit_button = ttk.Button(
            self.root,
            text = 'Edit Receiver Email',
            width = 20,
            command = self.edit_file
        )
        self.login_button = ttk.Button(
            self.root,
            text = 'Login Email',
            width = 20,
            command = self._login
        )
        self.send_button = ttk.Button(
            self.root,
            text = 'Send Email',
            width = 20,
            command = self.send_email
        )
        self.stop_button = ttk.Button(
            self.root,
            text = 'Stop',
            width = 20,
            command = self.stop_predict
        )
        self.reset_button = ttk.Button(
            self.root,
            text = 'Reset',
            width=20,
            command = self.reset
        )

        self.WorkTimeText = ttk.Label(self.root, textvariable = self.WorkTimeStr)
        self.VideoTimeText = ttk.Label(self.root, textvariable = self.VideoTimeStr)
        self.CurrentDirText = ttk.Label(self.root, textvariable = self.CurrentDir)
        self.CurrentVideoText = ttk.Label(self.root, textvariable = self.CurrentVideo)
        self.CurrentStatusText = ttk.Label(self.root, textvariable = self.CurrentStatus)
        self.open_button.pack(expand = False)
        self.edit_button.pack(expand = False)
        self.send_button.pack(expand = False)
        self.login_button.pack(expand = False)
        self.run_button.pack(expand = False)
        self.stop_button.pack(expand = False)
        self.reset_button.pack(expand = False)
        self.WorkTimeText.pack(expand = False)
        self.VideoTimeText.pack(expand = False)
        self.CurrentDirText.pack(expand = False)
        self.CurrentVideoText.pack(expand = False)
        self.CurrentStatusText.pack(expand = False)

    def terminate(self):
        os.kill(self.pid, signal.SIGTERM)
        self.stop_event.set()
        self.root.destroy()

    def edit_file(self):
        top = Tk()
        top.geometry("500x500")
        EditWindow(top)
        top.mainloop()

    def SecondToStr(self, second):
        Hour = second // 3600
        Minute = second // 60 - Hour * 60
        Second = second % 60
        if Hour != 0:
            return str(Hour) + 'h ' + str(Minute) + 'm ' + str(Second) + 's'
        elif Minute != 0:
            return str(Minute) + 'm ' + str(Second) + 's'
        else:
            return str(Second) + 's'

    def add_time(self, VideoTime, WorkTime):
        self.CurrentVideoTime = self.CurrentVideoTime + VideoTime
        self.CurrentWorkTime = self.CurrentWorkTime + WorkTime
        self.WorkTimeStr.set('Current Work Time = ' + self.SecondToStr(self.CurrentWorkTime))
        self.VideoTimeStr.set('Current Video Time = ' + self.SecondToStr(self.CurrentVideoTime))

    def search_new_file(self):
        print('Starting to load Video ...')
        self.Running = True
        self.CurrentStatus.set('Current Status: Running')
        CurrentThread = threading.currentThread()

        # write log 
        log_file = open("log.txt", "a")

        while getattr(CurrentThread, "do_run", True):
            Files = os.listdir(self.DirPath)
            for file in Files:
                if self.Running and (not file.startswith('(done)')) and file.endswith('.mp4'):
                    FilePath = self.DirPath + '/' + file
                    self.CurrentVideo.set('Current Predict Video: ' + str(file))
                    print('Current Predict Video: ' + str(file))
                    print(FilePath)
                    video = VideoConverter()
                    WorkTime, VideoTime = video.transform_and_predict(FilePath)

                    self.add_time(VideoTime, WorkTime)
                    print('----------------')
                    print('[VideoTime]', VideoTime)
                    print('[WorkTime]', WorkTime)
                    print('----------------')
                    os.rename(FilePath, self.DirPath + '/(done)' + file)
                    self.CurrentVideo.set('Current Predict Video: ')

                    if self.CurrentWorkTime >= self.work_thres:
                        print('----------------')
                        print('Exceeds 8 hours of work')
                        print('Send mail to boss!')
                        print('----------------')
                        log_file.write("'%s' video has exceeds 8 hours of working!!!(exceeded work time(seconds): %d)\n" % (str(file), self.CurrentWorkTime - self.work_thres))
                        
                        self.send_email()
                        self.is_over_work_thres = True

            time.sleep(1)

        log_file.close()

    def select_dir(self):
        DirName = filedialog.askdirectory(
            title = 'Open a directory',
            initialdir = self.DirPath
        )
        if DirName != '':
            self.DirPath = DirName
            self.CurrentDir.set('Current Directory: ' + self.DirPath)

    def run_predict_thread(self):
        if not self.PredictThread.is_alive():
            self.PredictThread.start()
        else:
            self.continue_predict()

    def reset(self):
        log_file = open("log.txt", "a")
        print(datetime.date.today(), 'Work Time =', self.SecondToStr(self.CurrentWorkTime), file = log_file)
        log_file.close()
        print('Reset Time')
        self.is_over_work_thres = False
        self.CurrentVideoTime = 0
        self.CurrentWorkTime = 0
        self.WorkTimeStr.set('Current Work Time = ' + self.SecondToStr(0))
        self.VideoTimeStr.set('Current Video Time = ' + self.SecondToStr(0))

    def reset_everyday(self):
        CurrentThread = threading.currentThread()
        while getattr(CurrentThread, "do_run", True):
            schedule.run_pending()
            time.sleep(1)

    def stop_predict(self):
        self.CurrentStatus.set('Current Status: Not Working')
        self.Running = False

    def continue_predict(self):
        self.CurrentStatus.set('Current Status: Running')
        self.Running = True

    def send_email(self):
        Receivers = open("EmailList.txt").readlines()
        for Recv in Receivers:
            print("Send to", Recv)
            SG = SendGmail(self.Account, self.Password, Recv)
            Failed = SG.send_message()
            if Failed:
                self.err = "failed"
            print("Failed =", Failed)

    def _login(self):
        self.CreateLoginWindow()

    def login(self):
        self.Account = self.account.get()
        self.Password = self.password.get()

        if not email_validate(self.Account):
            messagebox.showerror("Email Error", "Error: The email is invalid! Please try again")
        else:
            print(self.Account, self.Password)
            self.LoginWindow.destroy()

    def CreateLoginWindow(self):
        # ref: https://stackoverflow.com/questions/55560127/how-to-close-more-than-one-window-with-a-single-click
        self.LoginWindow = tk.Toplevel()
        self.LoginWindow.title("Wellcome to Login")
        self.LoginWindow.geometry('400x300')
        self.account = StringVar()
        self.account.set('your email address')
        self.password = StringVar()
        ttk.Label(self.LoginWindow, text = 'user:', font = ('Arial', 14)).place(x = 50, y = 85)
        ttk.Label(self.LoginWindow, text = 'password:', font = ('Arial', 14)).place(x = 50, y = 115)
        ttk.Entry(self.LoginWindow, textvariable = self.account, font = ('Arial', 14)).place(x = 150, y = 85)
        ttk.Entry(self.LoginWindow, textvariable = self.password, font = ('Arial', 14), show='*').place(x = 150, y = 115)
        ttk.Button(self.LoginWindow, text = 'Login', command = self.login).place(x = 120, y = 170)
    
    def start(self):
        print('Starting GUI ... ')

        # run the application
        self.ResetEveryDayThread.start()
        self.root.mainloop()
        self.PredictThread.do_run = False
        self.ResetEveryDayThread.do_run = False
        print('Finish')


class EditWindow(Frame):
    # http://hk.uwenku.com/question/p-hliiusca-tv.html
    def __init__(self, master = None):
        Frame.__init__(self, master)
        self.master = master
        self.master.title("Edit Eamil List")
        self.pack(fill=BOTH, expand=1)
        self.save_button = ttk.Button(
            self.master,
            text = 'Save',
            command=self.save_file
        )
        self.save_button.pack(expand=False)
        self.file_save = "EmailList.txt"

        self.text = Text(self.master, height=200, width=200)
        self.text.pack(side=LEFT, fill=Y, expand=True)

        self.scrollbar = Scrollbar(self.master, orient="vertical")
        self.scrollbar.config(command=self.text.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y, expand=True)

        # change all occurances of self.listNodes to self.text
        self.text.config(yscrollcommand=self.scrollbar.set)
        self.open_file()

    def open_file(self):
        self.text.delete("1.0", END)
        if os.path.isfile(self.file_save):
            with open(self.file_save, "r") as file:
                content = file.read()
                self.text.insert(END, content)

    def save_file(self):
        with open(self.file_save, 'w') as file:
            input = self.text.get("1.0", END)
            input = input[:-1]  # input.pop_back
            print(input, file=file, end='')

if __name__ == '__main__':
    app = GUI()
    app.start()