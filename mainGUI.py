import cv2
import os
import seg

from os import listdir
from tkinter import *
from tkinter import filedialog

class args:
    buffer = ""
    in_path = ""
    out_folder = ""

def selectFile():
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("Text files",
                                                      "*.txt*"),
                                                     ("all files",
                                                      "*.*")))
    label_in.configure(text="File Opened: " + filename)
    if(label_in['text']=="File Opened: "):
        args.in_path = label_in['text'].split()[2]


def selectInFolder():
    filename = filedialog.askdirectory()
    label_in.configure(text="Input Directory: " + filename)
    if(label_in['text']!="Input Directory: "):
        args.in_path = label_in['text'].split()[2]


def selectOutFolder():
    filename = filedialog.askdirectory()
    label_out.configure(text="Output Directory: " + filename)
    if(label_in['text']!="Output Directory: "):
        args.out_folder = label_out['text'].split()[2]


def segment():
    paramters = [args.buffer, args.in_path, args.out_folder]
    seg.main(args)


window = Tk()
window.title('Segmentation')
window.geometry("500x500")
window.config(background="white")
label_in = Label(window,
                 text="File Explorer using Tkinter",
                 width=100, height=4,
                 fg="blue")
label_out = Label(window,
                  text="File Explorer using Tkinter",
                  width=100, height=4,
                  fg="blue")
in_file = Button(window,
                 text="Select Input Files",
                 command=selectFile)
in_folder = Button(window,
                   text="Select Input Files",
                   command=selectInFolder)
out_folder = Button(window,
                    text="Select Output Folder",
                    command=selectOutFolder)
confirm = Button(window,
                 text="Segment Images",
                 command=segment)
button_exit = Button(window,
                     text="Exit",
                     command=exit)
label_in.pack()
label_out.pack()
in_file.pack()
in_folder.pack()
out_folder.pack()
confirm.pack()
button_exit.pack()
window.mainloop()
