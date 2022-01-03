import os
import time
import threading
import tkinter as tk
from Capturevideotoimage import VideoConverter
from SendGmail import SendGmail
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

# /mnt/d/my_code/CCU-project-ActionAI

class GUI():
    def __init__(self):
        print('Initialize ...')
        self.root = tk.Tk()
        self.root.title('Overwork Detection')
        self.root.resizable(False, False)
        self.root.geometry('300x300')
        self.DirPath = 'C:/Users/'
        self.CurrentPredictFile = str()
        self.AWSAccount = str()
        self.AWSPassword = str()
        self.CurrentWorkTime = 0
        self.CurrentVideoTime = 0
        self.PredictThread = threading.Thread(target=self.search_new_file)
        self.open_button = ttk.Button(
            self.root,
            text='Open a Directory',
            width=20,
            command=self.select_dir
        )
        self.run_button = ttk.Button(
            self.root,
            text='Run',
            width=20,
            command=self.run_predict_thread
        )
        self.edit_button = ttk.Button(
            self.root,
            text='Edit Receiver Email',
            width=20,
            command=self.edit_file
        )
        self.login_button = ttk.Button(
            self.root,
            text='Login AWS Email',
            width=20,
            command=self.set_AWS
        )
        self.send_button = ttk.Button(
            self.root,
            text='Send Email',
            width=20,
            command=self.send_email
        )

    def edit_file(self):
        top = Tk()
        top.geometry("500x500")
        EditWindow(top)
        top.mainloop()

    def search_new_file(self):
        global DirPath, CurrentPredictFile, CurrentVideoTime, CurrentWorkTime
        print('Starting to load Video ...')
        CurrentThread = threading.currentThread()
        while getattr(CurrentThread, "do_run", True):
            Files = os.listdir(self.DirPath)
            for file in Files:
                if (not file.startswith('(done)')) and file.endswith('.mp4'):
                    FilePath = self.DirPath + '/' + file
                    CurrentPredictFile = file
                    print(FilePath)
                    video = VideoConverter()
                    VideoTime, WorkTime = video.transform_and_predict(FilePath)

                    self.CurrentVideoTime = self.CurrentVideoTime + VideoTime
                    self.CurrentWorkTime = self.CurrentWorkTime + WorkTime
                    print('----------------')
                    print('[VideoTime]', self.CurrentVideoTime)
                    print('[WorkTime]', self.CurrentWorkTime)
                    print('----------------')
                    os.rename(FilePath, self.DirPath + '/(done)' + file)

            time.sleep(1)

    def select_dir(self):
        DirName = filedialog.askdirectory(
            title='Open a directory',
            # initialdir = self.DirPath,
            initialdir='/mnt/c/Users/money/my-code/CCU-project-ActionAI',
        )
        if DirName != '':
            self.DirPath = DirName

    def run_predict_thread(self):
        if not self.PredictThread.is_alive():
            self.PredictThread.start()

    def send_email(self):
        Recv = open("EmailList.txt").read()
        SG = SendGmail("test88812345@gmail.com", "1234567a.", Recv)
        Failed = SG.send_message()
        print("Failed = ", Failed)

    def set_AWS(self):
        Login = LoginWindow()
        self.AWSAccount, self.AWSPassword = Login.run()

    def start(self):
        print('Starting GUI ... ')
        self.open_button.pack(expand=False)
        self.edit_button.pack(expand=False)
        self.send_button.pack(expand=False)
        self.login_button.pack(expand=False)
        self.run_button.pack(expand=False)
        # run the application
        self.root.mainloop()
        self.PredictThread.do_run = False
        print('Finish')


class EditWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.master.title("Edit Eamil List")
        self.pack(fill=BOTH, expand=1)
        self.save_button = ttk.Button(
            self.master,
            text='Save',
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


class LoginWindow():
    def __init__(self):
        self.root = Tk()
        self.root.title("Wellcome to Login")
        self.root.geometry('400x350')
        self.account = StringVar()
        self.account.set('xxxx@gmail.com')
        ttk.Label(self.root, text='user:', font=(
            'Arial', 14)).place(x=50, y=185)
        ttk.Label(self.root, text='password:', font=(
            'Arial', 14)).place(x=50, y=215)
        ttk.Entry(self.root, textvariable=self.account,
                  font=('Arial', 14)).place(x=150, y=185)
        self.password = StringVar()
        ttk.Entry(self.root, textvariable=self.password, font=(
            'Arial', 14), show='*').place(x=150, y=215)
        ttk.Button(self.root, text='Login',
                   command=self.login).place(x=120, y=270.)

    def login(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return (self.account.get(), self.password.get())


if __name__ == '__main__':
    app = GUI()
    app.start()
