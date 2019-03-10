import os
import tarfile
import sys
import csv
import subprocess
import pandas as pd
import numpy as np
import natsort as ns
import requests
from multiprocessing import Pool, freeze_support

URL = "http://portoalegre.andrew.cmu.edu:88/BLUED/location_001_dataset_0YX.tar"

GLOABL_START_TIME_1_12 = pd.to_datetime("2011/10/20 11:58:32.6234999999996440291")
GLOABL_START_TIME_14_16 = pd.to_datetime("2011/10/26 01:20:48.5454583333330178151")

"""
Downloads untars, unzips the BLUED dataset.
Stores data in /data/raw.
Follwed by the event extraction algorithm.
Often crashes due to Ram limitation of 40 GB.

Arguments:
- left (int): Amount of samples left of the event
- right (int): Amount of samples right of the event
Returns:
- None
"""


def download_preprocess(left, right):
        for i in range(1, 17):
            if i < 10:
                new_url = URL.replace('X', str(i)).replace('Y', '0')
            else:
                new_url = URL.replace('X', str(i)[1]).replace('Y', str(i)[0])

            file_name = new_url.split('/')[-1]
            file_name = os.path.join("data", "raw", file_name)

            # Download one folder of the dataset
            print ("Starting to download: " + file_name)
            r = requests.get(new_url)
            with open(file_name, "wb") as code:
                code.write(r.content)
            print ("Finished Downloading: " + file_name)

            # Untar folder and delets .tar after the process.
            untar(file_name, os.path.join("data", "raw"))

            folder_path = os.path.join("data", "raw", new_url.split('/')[-1].split('.')[0])

            # Unzips files and deletes them afer the process. Set processes to your CPU count.
            file_path = []
            for bz2 in os.listdir(folder_path):
                file_path.append(os.path.join(folder_path, bz2))

            # Set CPU number here.
            pool = Pool(processes=15)
            result = pool.map_async(unbz2, file_path)
            result.get()

            # Non-threaded version
            #for bz2 in os.listdir(folder_path):
            #    file_path = os.path.join(folder_path, bz2)
            #    unbz2(file_path)
            # --------------------

            # Start of the event extraction.
            print ("Starting preprocessing for: " + folder_path)
            concat_files_to_csv(B=right, A=left)

            command = "rm -r -f " + folder_path
            subprocess.check_call(command.split())


"""
Concats all .txt data file to a single Pandas-Dataframe.

Arguments:
- A (int): Amount of samples left of the event
- B (int): Amount of samples right of the event
Returns:
- None
"""


def concat_files_to_csv(A, B):
    column = ['X_Value', 'Current A', 'Current B', 'VoltageA']
    for location_folder in os.listdir(os.path.join("data", "raw")):
        if 'location_001_dataset_' in location_folder:
                file_folder = os.path.join("data", "raw", location_folder)
                df_list = []
                concat_df = pd.DataFrame()
                for data in ns.natsorted(os.listdir(file_folder), key=lambda y: y.lower()):
                        if 'location_001_ivdata_' in data:
                                print ("Reading datafile: " + data)
                                df = pd.read_csv(os.path.join(file_folder, data), header=None, skiprows=23, index_col=False)
                                df_list.append(df)

                concat_df = pd.concat(df_list, axis=0)
                concat_df.columns = column
                concat_df.index = np.arange(len(concat_df.index))
                get_events(concat_df, file_folder, location_folder, A, B)


"""
Provides the event extraction for the BLUD data set.
Stores the .csv event files in /data/preprocessed.

Reminder:
Last 3 events of 14 equivalent to first 3 in 15. Data is in 14.
Problem with reseting the X_Value in file location_001_ivdata_5056 in location_001_dataset_013.

Arguments:
- A (int): Amount of samples left of the event
- B (int): Amount of samples right of the event
- file_folder (String): Folder with the data.
- location_folder (String): Folder with the 16 location folders.
Returns:
- None
"""


def get_events(data, file_folder, location_folder, A, B):
    print ("Reading eventslist .csv for: " + location_folder)
    events = pd.read_csv(os.path.join(file_folder, "location_001_eventslist.txt"))

    print ("Starting to get timestamp windows.")
    for i, timestamp in enumerate(events['Timestamp']):

        # Search the insertion index for the current event
        if (int((location_folder.split('_'))[-1])) == 13:
                if pd.to_datetime(timestamp) < GLOABL_START_TIME_14_16:
                        time_delta_sec = (pd.to_datetime(timestamp) - GLOABL_START_TIME_1_12).total_seconds()
                        series = data['X_Value']
                        idx = series[series == 0.000000].index[0]
                        mid = np.searchsorted(series[:idx], time_delta_sec)
                        mid = mid[0]
                else:
                        time_delta_sec = (pd.to_datetime(timestamp) - GLOABL_START_TIME_14_16).total_seconds()
                        series = data['X_Value']
                        idx = series[series == 0.000000].index[0]
                        mid = np.searchsorted(series[idx:], time_delta_sec)
                        mid = mid[0] + len(series[:idx])

        elif pd.to_datetime(timestamp) < GLOABL_START_TIME_14_16:
                time_delta_sec = (pd.to_datetime(timestamp) - GLOABL_START_TIME_1_12).total_seconds()
                mid = np.searchsorted(data['X_Value'], time_delta_sec)
                mid = mid[0]

        else:
                time_delta_sec = (pd.to_datetime(timestamp) - GLOABL_START_TIME_14_16).total_seconds()
                mid = np.searchsorted(data['X_Value'], time_delta_sec)
                mid = mid[0]

        upper = mid + B
        lower = mid - A

        try:
                print ("Found index " + str(mid) + " for delta: " + str(time_delta_sec)
                       + " actual delta: " + str(data['X_Value'][mid]))
        except KeyError:
                print("Found index " + str(mid) + " with " + str(time_delta_sec) + " is the last element.")

        window = data[lower:upper]

        # Shifts event widnow to start with a new cycle
        if str(events['Phase'].loc[i]) == 'A':
                idx = np.where(np.diff(np.signbit(window['Current A'][:250])))[0][0]
        else:
                idx = np.where(np.diff(np.signbit(window['Current B'][:250])))[0][0]

        window = data[lower + idx: upper + idx]

        new_path = os.path.join("data", "preprocessed")
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        # Strange names because windows compatibility
        print ("Saving data window for: " + timestamp)
        tmp = timestamp.replace(' ', '_').replace('/', '-').replace(':', '+') + ',' + str(events['Label'].loc[i]) + ',' + str(events['Phase'].loc[i] + '.csv')

        window.to_csv(os.path.join(new_path, tmp))


"""
Untars and delets a .tar.
Reference: https://sukhbinder.wordpress.com/2014/03/06/untar-a-tar-file-with-python/

Arguments:
- file_path (String): Path of the file.
- dest_path (String): Destination path.
Returns:
- None
"""


def untar(file_path, dest_path):
    if (file_path.endswith(".tar")):
        tar = tarfile.open(file_path)
        tar.extractall(path=dest_path)
        tar.close()
        print ("Extracted: " + file_path)
        os.remove(file_path)
        print ("Removed: " + file_path)
    else:
        print ("Not a .tar file: " + file_path)


"""
Unzips and delets a bz2.

Arguments:
- file_path (String): Path of the file.
Returns:
- None
"""


def unbz2(file_path):
    if (file_path.endswith(".bz2")):
        command = "bzip2 -d " + file_path
        subprocess.check_call(command.split())
        print ("Extracted: " + file_path)
    else:
        print("Not a .bz2 file: " + file_path)


# Set the amount of samples left of the event and right of the event.
if __name__ == '__main__':
    download_preprocess(left=6400, right=12200)


