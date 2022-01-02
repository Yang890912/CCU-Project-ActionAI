from Capturevediotoimage import VedioConverter 
import tkinter as tk
import os
import time
import threading
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

# /mnt/c/Users/money/Downloads/CCU-project-ActionAI
DirPath = 'C:/Users/money/Videos/'
selected = set()
def search_new_file():
    global DirPath, selected
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        Files = os.listdir(DirPath)
        for file in Files:
            if file.endswith('.mp4') and file not in selected:
                FilePath = DirPath + '/' + file
                print(FilePath)
                # video = VedioConverter()
                # video.test_predict(FilePath, 15, 120)
                selected.add(file)

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

thread = threading.Thread(target = search_new_file)
def run():
    if not thread.is_alive():
        thread.start()

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
    thread.do_run = False

