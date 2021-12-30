from Capturevediotoimage import VedioConverter 


curr_work_time = 0
curr_vedio_time = 0

if __name__ == '__main__':
    test = VedioConverter()
    test.test_predict("./train_images/v2/Produce_0.mp4", 15, 120)

from Capturevediotoimage import VedioConverter 
import tkinter as tk
import os
import time
import threading
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

# /mnt/c/Users/money/Downloads/GUI_v2/CCU-project-ActionAI/videos/
DirPath = 'C:/Users/money/Videos/'
def search_new_file():
    global DirPath
    selected = set()
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        Files = os.listdir(DirPath)
        for file in Files:
            if file.endswith('.mp4') and file not in selected:
                FilePath = DirPath + '/' + file
                print(FilePath)
                video = VedioConverter()
                video.test_predict(FilePath, 15, 120)
                selected.add(file)
                print(file)

        time.sleep(1)

# create the root window
root = tk.Tk()
root.title('Tkinter Open File Dialog')
root.resizable(False, False)
root.geometry('300x150')


def select_dir():
    global DirPath
    DirName = fd.askdirectory(
        title ='Open a directory',
        initialdir = '/',
	)

    if DirName != '':
        # showinfo (
        #     title = 'Selected Directory',
        #     message = DirName
        # )
        DirPath = DirName

def run():
    t = threading.Thread(target = search_new_file)
    t.start()

# open button
open_button = ttk.Button(
    root,
    text = 'Open a Directory',
    command = select_dir
)

run_button = ttk.Button(
    root,
    text = 'Run',
    command = run
)


if __name__ == '__main__':
    print('starting ... ')
    open_button.pack(expand=True)
    run_button.pack(expand=True)
    # run the application
    root.mainloop()
    t.do_run = False

