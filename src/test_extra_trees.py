import my_extra_trees, plot
import os, csv, sys, subprocess
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

"""
Testing frame for the Extra Tree classifier.
"""

APPLIANCE_DICT={
    101: "Living Room Desk Lamp",
    102: "Living Room Tall Desk Lamp",
    103: "Garage Door",
    108: "Kitchen Aid Chopper",
    111: "Refrigerator",
    112: "Living Room A/V System",
    118: "Computer 1",
    120: "Laptop 1",
    123: "Basement Receiver/DVR/Blueray Player",
    127: "Air Compressor",
    128: "LCD Monitor 1",
    129: "TV",
    131: "Printer",
    132: "Hair Dryer",
    134: "Iron",
    135: "Living Room Empty Socket",
    140: "Monitor 2",
    147: "Backyard Lights",
    148: "Washroom Light",
    149: "Office lights",
    150: "Closet light",
    151: "Upstairs hallway light",
    152: "Hallway stairs light",
    153: "Kitchen hallway light",
    155: "Kitchen overhead light",
    156: "Bathroom upstairs lights",
    157: "Dining Room overhead light",
    158: "Bedroom lights",
    159: "Basement lights",
    204: "Circuit 4",
    207: "Circuit 7",
    209: "Circuit 9",
    210: "Circuit 10",
    211: "Circuit 11"}

RESULTS = pd.DataFrame(columns=['Window', 'Circuit', 'Features','Augument', 'Folds', 'Estimatiors', 'F1', 'Y Pred', 'Y Test'])


"""
Creates a array of feaures specified by the parameters.

Arguments:
- circuit (int): 0: both, 1: A 2: B 
- A (int): Samples left of the event.
- B (int): Samples right of the event.
- feature_array (Array): Features to extract.
- augument (Bool): Wether the data should be augmented or not.
- data_dir (String): Directory of the event files.
Returns:
- Touple:
    - feature_vector (Array): Computed features.
    - label_vector (Array):  Labels for features.
"""
def get_feature_frame(circuit, A, B, feature_array, augument, data_dir):

    extra_trees = my_extra_trees.MyClassifier()

    # Only using one eventwidow file length
    B = 6400 + B
    A = 6400 - A
    
    feature_vector = []
    label_vector = []

    bar = tqdm(range(len(os.listdir(data_dir))))

    for event in os.listdir(data_dir):

        bar.update(1)

        event_file = os.path.join(data_dir, event)
        
        circuit_char = event.split(',')[-1]
        device = event.split(',')[-2]
        
        if circuit_char == 'A.csv' and (circuit == 0 or circuit == 1) and int(device) < 300:

            df = pd.read_csv(event_file)

            # Cut out event windnow
            window = df[A:B]
            feature = extra_trees.compute_input_signal(voltage = window['VoltageA'], 
                                                       current = window['Current A'],
                                                       feature_array=feature_array)
            feature_vector.append(feature)
            label_vector.append(int(device))

            # Data augumentation
            if augument == True:

                # Left phase shift
                window = df[A+200:B+200]
                feature = extra_trees.compute_input_signal(voltage = window['VoltageA'], 
                                                            current = window['Current A'],
                                                            feature_array=feature_array)
                feature_vector.append(feature)
                label_vector.append(int(device))

                # Left half phase flip
                window = df[A+100:B+100]
                window.loc[:,'Current A'] *= -1
                feature = extra_trees.compute_input_signal(voltage = window['VoltageA'], 
                                                            current = window['Current A'],
                                                            feature_array=feature_array)
                feature_vector.append(feature)
                label_vector.append(int(device))

        elif circuit_char == 'B.csv' and (circuit == 0 or circuit == 2) and int(device) < 300:

            df = pd.read_csv(event_file)

            # Voltage B = Voltage A * -1
            df.loc[:,'VoltageA'] *= -1

            window = df[A:B]
            feature = extra_trees.compute_input_signal(voltage = window['VoltageA'], 
                                                       current = window['Current B'],
                                                       feature_array=feature_array)

            feature_vector.append(feature)
            label_vector.append(int(device))

            # Data augumentation           
            if augument == True:
                # Left phase shift
                window = df[A+200:B+200]
                feature = extra_trees.compute_input_signal(voltage = window['VoltageA'], 
                                                            current = window['Current B'],
                                                            feature_array=feature_array)
                feature_vector.append(feature)
                label_vector.append(int(device))

                # Left half phase flip
                window = df[A+100:B+100]
                window.loc[:,'Current B'] *= -1
                feature = extra_trees.compute_input_signal(voltage = window['VoltageA'], 
                                                            current = window['Current B'],
                                                            feature_array=feature_array)
                feature_vector.append(feature)
                label_vector.append(int(device))

    bar.close()

    return (np.asarray(feature_vector), np.asarray(label_vector))


"""
Actual testing frame.
Uses Stratified K-Fold.
Saves Confusion Matrix plots.

Arguments:
- circuit (int): 0: both, 1: A 2: B 
- A (int): Samples left of the event.
- B (int): Samples right of the event.
- feature_array (Array): Features to extract.
- augument (Bool): Wether the data should be augmented or not.
- data_dir (String): Directory of the event files.
Returns:
- None
"""
def k_fold_fit(circuit = 0, A = 3000, B = 3000, feature_array = [1,0,0,0,0], augument = False, data_dir = os.path.join("data", "preprocessed")):

    feature_frame = get_feature_frame(circuit=circuit, data_dir=data_dir, A=A, B=B, augument=augument, feature_array=feature_array)

    # Set number of estimaters and folds for Stratified K-Fold
    n_estimators_array = [10, 50, 100, 500]
    folds_array = [5, 10]

    for l in range(len(folds_array)):
        for j in range(len(n_estimators_array)):

            folds = folds_array[l]
            n_estimators = n_estimators_array[j]

            for i in range(1,2):

                message = ('Window: ' + str(A+B) + ' Circuit: ' + str(circuit) + ' Features: ' 
                        + str(feature_array) + ' Augument: ' + str(augument) + ' Folds: ' 
                        + str(folds) + ' Estimators: ' + str(n_estimators))

                print('Testing ExtraTrees: ' + message)

                X = feature_frame[0]
                y = feature_frame[1]

                skf = StratifiedKFold(n_splits=folds)
                skf.get_n_splits(X, y)

                tmp_f1 = 0
                tmp_pred = []
                tmp_test = []

                for train_index, test_index in skf.split(X, y):     

                    X_train = X[train_index]
                    X_test = X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    extra_trees = my_extra_trees.MyClassifier(n_estimators=n_estimators, max_features=None, n_jobs=-1)

                    extra_trees.fit(X_train, y_train)

                    y_pred = extra_trees.predict(X_test)
                    f1 = f1_score(y_test, y_pred, average='micro')

                    # Saving best f1 result and print confusion matrix
                    if f1 > tmp_f1:
                        tmp_f1 = f1
                        tmp_test = y_test
                        tmp_pred = y_pred

                # Safing results in Dataframe 
                RESULTS.loc[len(RESULTS)] = [A+B, circuit, feature_array, augument, folds, n_estimators, tmp_f1, tmp_pred, tmp_test]

                # Plotting Confusion-Matrix
                file_name = (str(A+B) + '_' + str(circuit) + '_' + str(feature_array) + '_' 
                            + str(augument) + '_' + str(folds) + '_' + str(n_estimators))

                __plot_confusion_matrix(tmp_test, tmp_pred, file_name + '_' + str(tmp_f1) + '_' + '.svg')

"""
Saves te confusion matrix with the help of the plotting function.

Argumetns:
- y_test (Array): Array of tested labels.
- y_pred (Array): Array of predicted labels.
Returns:
- None
"""
def __plot_confusion_matrix(y_test, y_pred, name):
    classes = y_pred.tolist() + y_test.tolist()
    classes = list(set([x for x in classes if classes.count(x) >= 1]))
    classes = sorted(classes)

    cm = confusion_matrix(y_test, y_pred)

    plot.plot_confusion_matrix(cm=cm, classes=classes, title='Confusion matrix, without normalization')
    
    name = os.path.join("results", "plots_trees", name)

    plt.savefig(name)


if __name__ == '__main__':

    # Set features to use for evaluation 
    # 1: DCS 2: COT 3: AOT 4: AC 5:PS
    features = [[1,0,0,0,0], [1,0,0,1,0], [1,1,1,1,0], [0,1,1,1,0], [0,0,0,1,0]]

 
    for k in range(len(features)):

        k_fold_fit(circuit = 0, A = 6000, B = 12000, augument=True, feature_array=features[k])
        k_fold_fit(circuit = 1, A = 6000, B = 12000, augument=True, feature_array=features[k])
        k_fold_fit(circuit = 2, A = 6000, B = 12000, augument=True, feature_array=features[k])

        k_fold_fit(circuit = 0, A = 6000, B = 12000, augument=False, feature_array=features[k])
        k_fold_fit(circuit = 1, A = 6000, B = 12000, augument=False, feature_array=features[k])
        k_fold_fit(circuit = 2, A = 6000, B = 12000, augument=False, feature_array=features[k])

   
        k_fold_fit(circuit = 0, A = 6000, B = 6000, augument=True, feature_array=features[k])
        k_fold_fit(circuit = 1, A = 6000, B = 6000, augument=True, feature_array=features[k])
        k_fold_fit(circuit = 2, A = 6000, B = 6000, augument=True, feature_array=features[k])

        k_fold_fit(circuit = 0, A = 6000, B = 6000, augument=False, feature_array=features[k])
        k_fold_fit(circuit = 1, A = 6000, B = 6000, augument=False, feature_array=features[k])
        k_fold_fit(circuit = 2, A = 6000, B = 6000, augument=False, feature_array=features[k])


        k_fold_fit(circuit = 0, A = 3000, B = 3000, augument=True, feature_array=features[k])
        k_fold_fit(circuit = 1, A = 3000, B = 3000, augument=True, feature_array=features[k])
        k_fold_fit(circuit = 2, A = 3000, B = 3000, augument=True, feature_array=features[k])

        k_fold_fit(circuit = 0, A = 3000, B = 3000, augument=False, feature_array=features[k])
        k_fold_fit(circuit = 1, A = 3000, B = 3000, augument=False, feature_array=features[k])
        k_fold_fit(circuit = 2, A = 3000, B = 3000, augument=False, feature_array=features[k])

        # Stores result per feature in a pickle file
        path = os.path.join("results", "pkl_trees")
        RESULTS.to_pickle(path + 'tree_' + str(features[k]) + '.pkl')

