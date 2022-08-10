import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import scipy.io as spio
import os


path = "G:\My Drive\Lab\Video_analyiz\Amichai_Lavi_Data\Test\A_4_F1_e1"
os.chdir(path)

vidcap = cv2.VideoCapture(r"A-4 F1 e1.avi")
success, image = vidcap.read()
fps = vidcap.get(cv2.CAP_PROP_FPS)
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

dfh5 = pd.read_hdf(r'A-4 F1 e1DLC_resnet50_mouse_poseJul25shuffle1_30000.h5')

# Clac Center of mass
row = dfh5.iloc[[2]]
row = np.squeeze(row.to_numpy())
nose = row[:2]
ear = row[12:14]
center_of_mass = np.mean([nose, ear], axis=0)

# ## Visual inspection
# plt.imshow(image)
# plt.plot(nose[0], nose[1], 'bo')
# plt.plot(ear[0], ear[1], 'bo')
# plt.plot(center_of_mass[0], center_of_mass[1], 'r+')
# plt.show()

# Subtract CoM
dfh5.iloc[:, [0, 3, 6, 9, 12]] -= center_of_mass[0]
dfh5.iloc[:, [1, 4, 7, 10, 13]] -= center_of_mass[1]

# ## Visual inspection
# row = dfh5.iloc[[100]]
# row = np.squeeze(row.to_numpy())
# nose = row[:2]
# ear = row[12:14]
# plt.imshow(image)
# plt.plot(nose[0], nose[1], 'bo')
# plt.plot(ear[0], ear[1], 'bo')
# plt.plot(center_of_mass[0]-center_of_mass[0], center_of_mass[1]-center_of_mass[1], 'r+')
# plt.show()

# convert to np
pos_df = dfh5.iloc[1:, [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]]
pos = pos_df.to_numpy()

# pupil
dfh5_pupil = pd.read_hdf(
    r'Binary_A-4 F1 e1DLC_resnet50_pupil_dilationJul28shuffle1_40000.h5')

dist = lambda x, y: np.linalg.norm(x - y)

pupil = dfh5_pupil.iloc[1:, [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]]
pupil = pupil.to_numpy()
# i_up = pupil[:, 0:2]
# i_down = pupil[:, 2:4]
# i_right = pupil[:, 4:6]
# i_left = pupil[:, 6:8]
dist_hor = list(map(dist, pupil[:, 0:2], pupil[:, 4:6]))
dist_ver = list(map(dist, pupil[:, 4:6], pupil[:, 6:8]))

pos_n_pupil = np.concatenate((pos, pupil), axis=1)
#
# # Visual inspection
# plt.plot(time, dist_ver, label='Vertical Distance')
# plt.plot(time, dist_hor, label='Horizontal Distance')
# plt.legend()
# plt.show()
#
# plt.figure()
# plt.imshow(image)
# plt.plot(i_up[10, 0], i_up[10, 1], 'b+')
# plt.plot(i_down[10, 0], i_down[10, 1], 'bo')
# plt.plot(i_right[10, 0], i_right[10, 1], 'r+')
# plt.plot(i_left[10, 0], i_left[10, 1], 'ro')
# plt.show()


###
# get session data
data = spio.loadmat('data')

## visual inspection
# Go_val = np.ones_like(data['Go_times'])
# NoGo_val = np.ones_like(data['NoGo_times'])*0.8
# Licks_val = np.ones_like(data['Lick_times'])*0.6
#
# plt.scatter( data['Go_times'], Go_val, label='Go', marker='|', lw=3, color='green')
# plt.scatter(data['NoGo_times'], NoGo_val, label='NoGo', marker='|', lw=3, color='red')
# plt.scatter(data['Lick_times'], Licks_val, label='Licks', marker='|', lw=0.1, color='black')
# plt.legend()
# plt.xlabel("Time [sec]")
# plt.show()

stimuli = np.sort(np.append(data['Go_times'], data['NoGo_times']))

# convert to the right timescale
stimuli = stimuli * fps

# divide into segments according to stimulus
seg_data = np.split(pos_n_pupil, (stimuli.astype(int)))

# Correct the times (for this specific subject)
a = np.zeros([int((np.max(data['Time']) - duration) * fps), pos_n_pupil.shape[1]])
a = np.concatenate((a, pos_n_pupil), axis=0)
seg_data = np.split(a, (stimuli.astype(int)))


rel_time = int(2 * fps)
for i in range(len(stimuli) + 1):
    seg_data[i] = seg_data[i][-rel_time:, :]

seg_data = np.array(seg_data)

# Get licks in time
lick_seg = np.zeros(len(stimuli) + 1)
t = np.arange(int(np.max(data['Time']) * fps))
seg_time = np.split(t, (stimuli.astype(int)))

i = 0
for tLick in np.ravel(data['Lick_times'] * fps):

    if (i < len(stimuli)) and (tLick > stimuli[i]):
        i += 1
    if np.isin(int(tLick), seg_time[i][:rel_time]):
        lick_seg[i] += 1

# PCA
X = np.reshape(seg_data, [51, rel_time * 20])
X = X[5:, :]
y = np.where(lick_seg > 0, 1, 0)
y = y[5:]

# share the results
X = np.c_[X, y]

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# Training and Making Predictions
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)



# Evaluating the Performance
from sklearn.metrics import accuracy_score
print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))


# k-fold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    # if ylim is not None:
    #     axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 2, figsize=(10, 15))
title = "Learning Curves (Naive Bayes)"
# Cross validation with 50 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(
    estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4
)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(
    estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4
)

plt.show()