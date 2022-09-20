import numpy as np
import pandas as pd
import cv2
import scipy.io as spio
import os
import re
import glob

def get_Features(dir, i):
    dist = lambda x, y: np.linalg.norm(x - y)

    movs = glob.glob(os.path.join(dir[i] + "/*.avi"))
    vidcap = cv2.VideoCapture(movs[0])
    success, image = vidcap.read()
    print("Able to read movie?: " + str(success))
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    # load facial points
    dfh5_file = glob.glob(os.path.join(dir[i] + "/*30000.h5"))
    dfh5 = pd.read_hdf(dfh5_file[0])

    # Clac Center of mass
    row = dfh5.iloc[[2]]
    row = np.squeeze(row.to_numpy())
    nose = row[:2]
    ear = row[12:14]
    center_of_mass = np.mean([nose, ear], axis=0)

    # Subtract CoM
    dfh5.iloc[:, [0, 3, 6, 9, 12]] -= center_of_mass[0]
    dfh5.iloc[:, [1, 4, 7, 10, 13]] -= center_of_mass[1]

    # convert to np
    pos_df = dfh5.iloc[1:, [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]]

    pos = pos_df.to_numpy()
    com = np.ones_like(pos[:, 0:1]) * center_of_mass
    # nose = pos[:,:2] | i_up = pos[:,2:4] | i_down = pos[:,4:6] | leg = [:,6:8] | leg = [:,8:10]
    nose = list(map(dist, pos[:, 0:2], com))
    ear = list(map(dist, pos[:, 6:8], com))
    leg = list(map(dist, pos[:, 8:10], com))
    pos = np.c_[nose, ear, leg]

    # pupil
    dfh5_pupil_file = glob.glob(os.path.join(dir[i] + "/*70000.h5"))
    if dfh5_pupil_file:
        dfh5_pupil = pd.read_hdf(dfh5_pupil_file[0])
        dfh5_pupil.shape

        pupil = dfh5_pupil.iloc[1:, [0, 1, 3, 4, 6, 7, 9, 10]]
        pupil = pupil.to_numpy()

        # i_up = pupil[:, 0:2] |  i_down = pupil[:, 2:4]  |  i_right = pupil[:, 4:6] |  i_left = pupil[:, 6:8]
        dist_hor = list(map(dist, pupil[:, 0:2], pupil[:, 2:4]))
        dist_ver = list(map(dist, pupil[:, 4:6], pupil[:, 6:8]))

        pos_n_pupil = np.c_[pos, dist_hor, dist_ver]


    else:
        print("--The mouse have no pupil file")
        pos_n_pupil = pos

    return pos_n_pupil, pos_df, fps


def getData(dir, before_q=2, after_q=2):
    df = pd.DataFrame(
        columns=['Name', 'Task Level', 'Lick Rate [%]', 'True Positive', 'False Positive', 'Total Accuracy', 'Features',
                 'Labels'])

    for i in range(dir.__len__()):
        print("\nMouse " + str(i + 1))
        print(dir[i])
        print("------------------")

        file = os.path.normpath(dir[i]).split(os.path.sep)

        pos_n_pupil, pos_df, fps = get_Features(dir, i)

        # run matlab code if necessary ("Z:\Shared\Amichai\importfile.m")

        # get session data
        data = spio.loadmat(os.path.join(dir[i] + 'data.mat'))
        correction = pos_n_pupil.shape[0] / ((data['time'].shape[1] / data['Fs']) * fps)
        tSOS_new = data['tSOS'] * correction
        Lick_times_new = np.ravel(data['Lick_times'] * correction + (
                tSOS_new[0] - np.minimum(data['Go_times'][0, 0], data['NoGo_times'][0, 0])))

        stimuli = np.ravel(np.sort(np.c_[data['Go_times'], data['NoGo_times']]))
        stimuli_type = np.in1d(stimuli, data['Go_times'])

        if len(stimuli) != len(tSOS_new):
            print("--Can't label the cue type\n **SKIP** \n ")
            continue

        # movie interpolation
        r = pd.RangeIndex(0, int((data['time'].shape[1] / data['Fs']) * fps), 1)
        t = pd.DataFrame(pos_n_pupil)
        t = t.sort_index()
        new_idx = np.linspace(t.index[0], len(r), len(r))
        t = (t.reindex(new_idx, method='ffill', limit=1).iloc[1:].interpolate())

        pos_n_pupil = t.to_numpy()

        # segment to trails
        Lick_times = Lick_times_new * fps
        for idx, seg in enumerate(tSOS_new * fps):
            segment = pos_n_pupil[int(seg - before_q * fps):int(seg), :].reshape(
                [int(before_q * fps) * pos_n_pupil.shape[1]])
            lick_seg = np.any(
                np.ravel(np.where((Lick_times > seg) & (Lick_times < seg + after_q * fps))))

            if idx == 0:
                prv_lick = np.array([0])  # previous segment lick choice as a feature
                x = np.r_[segment, prv_lick]
                y = lick_seg


            else:
                y = np.row_stack((y, lick_seg))
                prv_lick = y[-2]  # previous lick choice
                segment = np.r_[segment, prv_lick]
                x = np.row_stack((x, segment))

        lick_bias = y.sum() / y.shape[0]

        if lick_bias < 0.45 or lick_bias > 0.75:
            print("**Biased Licker**")
            continue

        level = np.full(y.shape, [re.findall("[he]", file[-1])])
        name = np.full(y.shape, [re.findall("A.._", file[-1])])
        # concat this mouse data to the others
        # create empty arrays
        if i == 0:
            X = x
            print(X.shape)

            y_tot = y
            stimuli_tot = stimuli_type
            level_tot = level
            name_tot = name


        else:
            X = np.concatenate((X, x), axis=0)
            print(X.shape)
            y_tot = np.concatenate((y_tot, y), axis=0)
            stimuli_tot = np.concatenate((stimuli_tot, stimuli_type), axis=0)
            level_tot = np.concatenate((level_tot, level), axis=0)
            name_tot = np.concatenate((name_tot, name), axis=0)

        TP = len(data['Reward_times'][0]) / len(data['Go_times'][0])
        try:
            FP = len(data['Punishment_times'][0]) / len(data['NoGo_times'][0])
        except:
            FP = 0
        TN = 1 - FP
        FN = 1 - TP
        Acc = (TP + TN) / (TP + TN + FP + FN)

        mouse = pd.DataFrame({
            'Name': re.findall("A.._", file[-1]),
            'Task Level': re.findall("[he]", file[-1]),
            'Lick Rate [%]': lick_bias * 100,
            'True Positive': TP,
            'False Positive': FP,
            'Total Accuracy': Acc,
            'Features': [x],
            'Labels': [y]}, )
        df = pd.concat([df, mouse], ignore_index=True)
    return X, np.ravel(y_tot), np.ravel(stimuli_tot), df, level_tot, name_tot


def get_unfilttered_Features(dir, i):
    movs = glob.glob(os.path.join(dir[i] + "/*.avi"))
    vidcap = cv2.VideoCapture(movs[0])
    success, image = vidcap.read()
    print("Able to read movie?: " + str(success))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # load facial points
    dfh5_file = glob.glob(os.path.join(dir[i] + "/*30000.h5"))
    dfh5 = pd.read_hdf(dfh5_file[0])

    # convert to np
    pos_df = dfh5.iloc[1:, [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]]
    pos = pos_df.to_numpy()

    # pupil
    dfh5_pupil_file = glob.glob(os.path.join(dir[i] + "/*70000.h5"))
    if dfh5_pupil_file:
        dfh5_pupil = pd.read_hdf(dfh5_pupil_file[0])

        dist = lambda x, y: np.linalg.norm(x - y)

        pupil = dfh5_pupil.iloc[1:, [0, 1, 3, 4, 6, 7, 9, 10]]
        pupil = pupil.to_numpy()

        # i_up = pupil[:, 0:2] |  i_down = pupil[:, 2:4]  |  i_right = pupil[:, 4:6] |  i_left = pupil[:, 6:8]
        dist_hor = list(map(dist, pupil[:, 0:2], pupil[:, 4:6]))
        dist_ver = list(map(dist, pupil[:, 4:6], pupil[:, 6:8]))

        pos_n_pupil = np.c_[pos, dist_hor, dist_ver]

    return pos_n_pupil, pos_df, fps
