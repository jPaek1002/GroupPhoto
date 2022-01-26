import seg

from tkinter import *
from tkinter import filedialog

class args:
    buffer = ""
    in_path = ""
    out_folder = ""
    div = 1

def select_in_folder():
    filename = filedialog.askdirectory()
    label_in.configure(text="Input Directory: " + filename)
    if(label_in['text']!="Input Directory: "):
        args.in_path = label_in['text'].split()[2]

def select_in_video():
    filename = filedialog.askopenfilename(filetypes=(("Video files", "*.mp4;*.flv;*.avi;*.mkv,*.mov,*.MOV"),
                                       ("All files", "*.*") ))
    label_in.configure(text="Input Directory: " + filename)
    if(label_in['text']!="Input Directory: "):
        args.in_path = label_in['text'].split()[2]
def select_out_folder():
    filename = filedialog.askdirectory()
    label_out.configure(text="Output Directory: " + filename)
    if(label_in['text']!="Output Directory: "):
        args.out_folder = label_out['text'].split()[2]

def segment():
    set_div()
    seg.main(args)


window = Tk()
window.title('Segmentation')
window.geometry("500x300")
window.config(background="white")
label_in = Label(window,
                 text="Input Directory: ",
                 width=100, height=4,
                 fg="blue")
label_out = Label(window,
                  text="Output Directory: ",
                  width=100, height=4,
                  fg="blue")
in_folder = Button(window,
                   text="Select Input Files",
                   command=select_in_folder)
in_video = Button(window,
                   text="Select Input video",
                   command=select_in_video)
out_folder = Button(window,
                    text="Select Output Folder",
                    command=select_out_folder)
confirm = Button(window,
                 text="Segment Images",
                 command=segment)
button_exit = Button(window,
                     text="Exit",
                     command=exit)
slider = Scale(window,
               from_=1,
               to=24,
               orient=HORIZONTAL)
label_in.pack()
label_out.pack()
in_video.pack()
slider.pack()
in_folder.pack()
out_folder.pack()
confirm.pack()
button_exit.pack()

def set_div():
    args.div = slider.get()
    
if __name__ == "__main__":
    window.mainloop()
