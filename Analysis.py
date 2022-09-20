def analysis(path, k=100):
    import numpy as np
    import pandas as pd
    import os
    from cls_svm import cls_svm

    os.chdir(path)

    data = np.load('Features_n_label.npz')
    X = data['X']
    Y = data['y']
    level = np.ravel(data['level'])
    name = np.ravel((data['name']))
    Stimuli_type = data['stimuli_type']

    level_name = np.unique(level)
    mice_name = np.unique(name)
    add_prev_q = [True, False]
    add_q = [True, False]
    K = k


    results = pd.DataFrame(columns=["Level", "Mouse", "Train_acc", "Train std",
                                    "Test_acc", "Test std",
                                    "Lick Bias", "Cue added?", "Previous Cue"])
    for q in add_q:
        for prev_q in add_prev_q:
            for lev in level_name:
                print("\nLevel: " + str(lev))
                print("________________________")
                lev_idx = np.where(level == lev)
                x = X[lev_idx]
                y = Y[lev_idx]
                stimuli_type = Stimuli_type[lev_idx]

                train_acc, test_acc, lick_bias, train_std, test_std = cls_svm(X=x, y=y,
                                                                              stimuli_type=stimuli_type,
                                                                              add_prev_q=prev_q,
                                                                              add_q=q, k=K)

                result = pd.DataFrame({"Level": lev, "Mouse": -1, "Train_acc": train_acc,
                                       "Train std": train_std,
                                       "Test_acc": test_acc, "Test std": test_std, "Lick Bias": lick_bias,
                                       "Cue added?": q, "Previous Cue": prev_q}, index=[1])

                results = pd.concat([results, result], ignore_index=True)

    for q in add_q:
        print("\n\nAdd Q:" +str(q))
        for prev_q in add_prev_q:
            print("Add rev Q:" + str(prev_q))
            for mouse in mice_name:
                print("\nMouse name: " + str(mouse))
                for lev in level_name:

                    print("level: " + str(lev))
                    print("######")
                    mice_idx = np.where(name == mouse)
                    lev_idx = np.where(level == lev)
                    both = np.intersect1d(lev_idx, mice_idx)
                    if not both.any():
                        continue

                    x = X[both]
                    y = Y[both]
                    stimuli_type = Stimuli_type[both]

                    train_acc, test_acc, lick_bias, train_std, test_std = cls_svm(X=x, y=y,
                                                                                  stimuli_type=stimuli_type,
                                                                                  add_prev_q=prev_q,
                                                                                  add_q=q, k=K)

                    result = pd.DataFrame({"Level": lev, "Mouse": mouse, "Train_acc": train_acc,
                                           "Train std": train_std,
                                           "Test_acc": test_acc, "Test std": test_std, "Lick Bias": lick_bias,
                                           "Cue added?": q, "Previous Cue": prev_q}, index=[1])
                    results = pd.concat([results, result], ignore_index=True)

    results.to_csv("results_h_e_division.csv")
