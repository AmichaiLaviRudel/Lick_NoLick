def cls_svm(X, y, stimuli_type, add_prev_q, add_q, add_answer=False, k=1000):

    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
    from sklearn import svm

    # Add the cue type
    if add_prev_q:
        x = X
    else:
        x = X[:, :-1]

    if add_q:
        x = np.c_[x, stimuli_type]

    # Add the answer
    if add_answer:
        x = np.c_[x, y]

    K = k
    acc = np.zeros([K, 2])

    for k in range(K):
        print(str(k) + "/" + str(K), end="\r")
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Training and Making Predictions
        # classifier = GaussianNB()
        classifier = svm.SVC(kernel='rbf')
        # classifier = RandomForestClassifier(max_depth=2, random_state=0)

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        acc[k, 0] = accuracy_score(y_test, y_pred)
        train_pred = classifier.predict(X_train)
        acc[k, 1] = accuracy_score(y_train, train_pred)

    # Evaluating the Performance
    train_acc = np.mean(acc[:, 1])
    train_std = np.std(acc[:, 1])
    test_acc = np.mean(acc[:, 0])
    test_std = np.std(acc[:, 0])

    print('Train accuracy: ' + str(round(train_acc, 2)) + ' || STD: ' + str(round(train_std, 3)))
    print('Test accuracy: ' + str(round(test_acc, 2)) + ' || STD: ' + str(round(test_std, 2)))

    lick_bias = y.sum() / y.shape[0] * 100

    print(
        "---------------------------------------\nLick Events: ~" + str(
            round(lick_bias)) + "% || No-lick Events: ~" + str(
            round(100 - lick_bias)) + "%")


    return train_acc, test_acc, lick_bias, train_std, test_std