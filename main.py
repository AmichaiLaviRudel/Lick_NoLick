from Analysis import analysis
from functions import *
from tkinter import Tk, filedialog
import glob

run_features = True
run_classifier = True

############################################################################

# Choose a folder
root = Tk()  # pointing root to Tk() to use it as Tk() in program.
root.withdraw()  # Hides small tkinter window.
root.attributes('-topmost', True)  # Opened windows will be active. above all windows despite selection.
path = filedialog.askdirectory()  # Returns opened path as str
dir = glob.glob(os.path.join(path, "*", ""), recursive=True)

os.chdir(path)
###########################################################################


# Check if the data set is compleat
for i in range(dir.__len__()):
    file_name = glob.glob(os.path.join(dir[i] + "/*70000.h5"))
    if file_name == []:
        print("missing pupil:")
        print(dir[i])
        print("Run DEEPLABCUT on the hive computer")
for i in range(dir.__len__()):
    file_name = glob.glob(os.path.join(dir[i] + "/*30000.h5"))
    if file_name == []:
        print("missing pos:")
        print(dir[i])
        print("Run DEEPLABCUT on the hive computer")

##########################################################################
# Get Features

if run_features:
    X, y, stimuli_type, mice_data, level_tot, name_tot = getData(dir=dir, before_q=2, after_q=2)

    # Save the data
    mice_data.to_pickle('./Mice_Data.pkl')
    np.savez('./Features_n_label', X=X, y=y, stimuli_type=stimuli_type, level=level_tot, name=name_tot)

##########################################################################

if run_classifier:
    analysis(path)

##########################################################################


