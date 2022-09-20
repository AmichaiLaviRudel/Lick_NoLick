from functions import *
import numpy as np
import pandas as pd
import scipy.io as spio
import os
import re
from tkinter import Tk, filedialog
import glob
import cv2


# Create features data frame for a specific subject
root = Tk()  # pointing root to Tk() to use it as Tk() in program.
root.withdraw()  # Hides small tkinter window.
root.attributes('-topmost', True)  # Opened windows will be active. above all windows despite selection.
path = filedialog.askdirectory()  # Returns opened path as str

# Choose a folder
dir = glob.glob(os.path.join(path, "*", ""), recursive=True)

before_q = 2
after_q = 2

for i in range(dir.__len__()):
    print("\nMouse " + str(i + 1))
    print(dir[i])
    print("------------------")

    file = os.path.normpath(dir[i]).split(os.path.sep)

    pos_n_pupil, pos_df, fps = get_Features(dir, i)

    # get session data
    data = spio.loadmat(os.path.join(dir[i] + 'data.mat'))

    correction = pos_n_pupil.shape[0] / ((data['time'].shape[1] / data['Fs']) * fps)

    tSOS_new = data['tSOS'] * correction
    Lick_times_new = np.ravel(data['Lick_times'] * correction + (
            tSOS_new[0] - np.minimum(data['Go_times'][0, 0], data['NoGo_times'][0, 0])))

    stimuli = np.ravel(np.sort(np.c_[data['Go_times'], data['NoGo_times']]))
    stimuli_type = np.in1d(stimuli, data['Go_times'])

    # movie interpolation
    r = pd.RangeIndex(0, int((data['time'].shape[1] / data['Fs']) * fps), 1)
    t = pd.DataFrame(pos_n_pupil)
    t = t.sort_index()
    new_idx = np.linspace(t.index[0], len(r), len(r))
    t = (t.reindex(new_idx, method='ffill', limit=1).iloc[1:].interpolate())

    pos_n_pupil = t.to_numpy()

    if len(stimuli) != len(tSOS_new):
        print("--Can't label the cue type\n **SKIP** \n ")
        continue

    # segment to trails
    Lick_times = Lick_times_new * fps
    for idx, seg in enumerate(tSOS_new * fps):
        segment = pos_n_pupil[int(seg - before_q * fps):int(seg + after_q * fps), :]
        lick_seg = np.any(
            np.ravel(np.where((Lick_times > seg) & (Lick_times < seg + after_q * fps))))

        if idx == 0:
            x = np.r_[segment]
            y = lick_seg

        else:
            x = np.row_stack((x, segment))
            y = np.row_stack((y, lick_seg))

    x = np.reshape(x, (len(tSOS_new), segment.shape[0], segment.shape[1]))

    level = np.full(x.shape[0], [re.findall("[he]", file[-1])])
    name = np.full(x.shape[0], [re.findall("A.._", file[-1])])


    # concat this mouse data to the others
    # create empty arrays
    if i == 0:
        X = x
        y_tot = y
        stimuli_tot = stimuli_type
        level_tot = level
        name_tot = name

    else:
        X = np.concatenate((X, x), axis=0)
        y_tot = np.concatenate((y_tot, y), axis=0)

        print(y_tot.shape)
        stimuli_tot = np.concatenate((stimuli_tot, stimuli_type), axis=0)
        level_tot = np.concatenate((level_tot, level), axis=0)
        name_tot = np.concatenate((name_tot, name), axis=0)

os.chdir(path)
np.savez('./segmented_Features_n_label', X=X, y=y_tot, stimuli_type=stimuli_tot, level=level_tot, name=name_tot)