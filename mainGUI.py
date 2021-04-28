import cv2
import os
import seg

from os import listdir
from tkinter import *
from tkinter import filedialog

in_directory = ""
out_directory = ""

def selectFile():
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("Text files",
                                                      "*.txt*"),
                                                     ("all files",
                                                      "*.*")))
    label_file_explorer.configure(text="File Opened: " + filename)

def selectFolder():
    filename = filedialog.askdirectory()
    label_file_explorer.configure(text="Output Directory: " + filename)

if __name__ == "__main__":
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
                         command=selectFolder)
    out_folder = Button(window,
                            text="Select Output Folder",
                            command=selectFolder)
    button_exit = Button(window,
                         text="Exit",
                         command=exit)
    label_in.pack()
    label_out.pack()
    in_file.pack()
    in_folder.pack()
    out_folder.pack()
    button_exit.pack()
    window.mainloop()