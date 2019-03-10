import pandas as pd
import numpy as np 
import features
from sklearn.ensemble import ExtraTreesClassifier



"""
Reference implementation for the following Appliance Identification algorithm:
Current Peak based Device Classification in NILM on a low-cost Embedded platform using Extra-Trees
By author: Aashish Kumar Jain et al.
Link to paper: https://ieeexplore.ieee.org/document/8284200/
The algorithm implementation follows the sklearn API
Short Description of the algorithm:
-Input: Voltage and Current
1. Preprocess the signal:
1.1 Extract current peaks of the event window
1.2 Divide into A and B
1.2 Build the mean of A
1.2 Subtract the mean of A of B (DCS)
2. Training the classifier
3. Classification
"""
class MyClassifier(ExtraTreesClassifier):

    def compute_input_signal(self, current, voltage, feature_array=[1,0,0,0,0]):

        dcs = []; ac = []; cot = []; aot = []; ps = []

        if feature_array[0] == 1:
            dcs = features.dcs(current)

        if feature_array[1] == 1:
            cot = features.cot(current)

        if feature_array[2] == 1:
            aot = features.aot(current, voltage)
            
        if feature_array[3] == 1:
            ac = features.ac_power(current, voltage)

        if feature_array[4] == 1:
            ps = features.phase_shift(current, voltage)


        feature_vector = np.concatenate((dcs, ac, cot, aot, ps))

        return np.asarray(feature_vector)
        


   