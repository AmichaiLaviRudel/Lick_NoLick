import cv2
import matplotlib.pyplot as plt
import scipy.io as spio
import os
import matplotlib
import time


matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# %matplotlib ipympl# Create features data frame for a specific subject
import numpy as np
from tkinter import Tk, filedialog
import glob


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


# %% choose a folder
root = Tk()  # pointing root to Tk() to use it as Tk() in program.
root.withdraw()  # Hides small tkinter window.
root.attributes('-topmost', True)  # Opened windows will be active. above all windows despite selection.
path = filedialog.askdirectory()  # Returns opened path as str
dir = glob.glob(os.path.join(path, "*", ""), recursive=True)

for i in range(dir.__len__()):    # Read the movie
    movs = glob.glob(os.path.join(dir[i] + "/*.avi"))
    print(movs)
    if len(movs) > 1:
        print("Already exists\n**SKIP**")
        continue

    vidcap = cv2.VideoCapture(movs[0])
    frame_width = int(vidcap.get(3))

    frame_height = int(vidcap.get(4))
    file = os.path.normpath(dir[i]).split(os.path.sep)
    file_name = os.path.join(dir[i] +file[-1]+ '_binary.avi')

    out = cv2.VideoWriter(file_name,  cv2.VideoWriter_fourcc(*"MJPG"), 30, (frame_width, frame_height))

    if vidcap.isOpened() == False:
        print("Error")
    l = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

    while vidcap.isOpened():
        success, img = vidcap.read()
        if not success:
            break

        (thresh, blackAndWhiteImage) = cv2.threshold(img, 82,82, cv2.THRESH_BINARY_INV)
        out.write(blackAndWhiteImage)

        count += 1
        printProgressBar(count + 1, l, prefix='Progress:', suffix='Complete', length=50)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    vidcap.release()
    out.release()
    cv2.destroyAllWindows()
