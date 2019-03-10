import pandas as pd
import numpy as np 
import os, csv, sys
import features

# Dictionary used to discard outliners
MEAN_DICT = {
    101: 8.756693501430021,
    102: 8.490014426841661,
    103: 7.512500249205273,
    108: 10.714342781715489,
    111: 6.917252612229534,
    112: 9.858161512261333,
    118: 8.430979346215203,
    120: 9.972306361481811,
    123: 16.15348959106992,
    127: 7.60520106251382,
    128: 8.76975743638495,
    129: 6.9630452500116755,
    131: 8.838286429791058,
    132: 9.747753054692224,
    134: 8.535963918382148,
    135: 0,
    140: 8.350523398054486,
    147: 7.098447546648604,
    148: 8.016804281971346,
    149: 8.627458089562985,
    150: 9.00997190670099,
    151: 9.097633442124556,
    152: 7.928089052434642,
    153: 10.69495417201538,
    155: 8.409828355662714,
    156: 7.425518319384423,
    157: 8.332781152532055,
    158: 7.401750161088491,
    159: 8.20846162823607,
    204: 6.844930149256015,
    207: 7.374203490839471,
    209: 8.819260747815358,
    210: 9.542595886312407,
    211: 8.377703406721746}

"""
Reference implementation for the following Appliance Identification algorithm:
Event-Based Energy Disaggregation Algorithm for Activity Monitoring From a Single-Point Sensor
By author: José M. Alcalá et al.
Link to paper: https://ieeexplore.ieee.org/document/7997812
The algorithm implementation follows the sklearn API
Short Description of the algorithm:
-Input: Voltage and Current
1. Preprocess the signal:
1.1 Extract previous cycle of the event window
1.2 Subtract cylce of the event widnow
1.3 Calculate SPQD trajectroies / other features
2. Training the classifier
2.1 Compute Scatter Matrix
2.2 Dimension reduction
3. Classification
3.1 Compute recreation matrix
3.2 Compute vector norm
"""
class MyPCA():

    def __init__(self, n_components = 0.97, tresh_hold = 20):
        self.n_components = n_components
        self.tresh_hold = tresh_hold

        self.model = {}
        self.var_dict = {}

        self.explained_variance_ratio_sum = {}
        self.explained_variance_ratio_ = {}
        self.singular_values_= {}


    def compute_input_signal(self, current, voltage, feature_array=[1,0,0,0,0,0]):

        # Preprocess current as presented in the paper
        cycle_n = current.size / 200 # 61
        current_cycles = np.split(current, cycle_n)
                
        # y = {i_e-30, ... , i_e, I_e30}    (10)
        y = np.asarray(current_cycles[1:])

        # I_e = y - {I_e-31, ... , I_e-31}  (11)
        I_e = np.apply_along_axis(lambda x : np.subtract(x, current_cycles[0]), 1, y)

        current = np.ndarray.flatten(I_e) 
        voltage = voltage[200:]


        # Calculate features
        dcs = []; ac = []; cot = []; aot = []; ps = []; ac_p = []
        
        if feature_array[0] == 1:
            ac_p = self.__ac_power(current, voltage)

        if feature_array[1] == 1:
            dcs = features.dcs(current)

        if feature_array[2] == 1:
            cot = features.cot(current)

        if feature_array[3] == 1:
            aot = features.aot(current, voltage)
            
        if feature_array[4] == 1:
            ac = features.ac_power(current, voltage)

        if feature_array[5] == 1:
            ps = features.phase_shift(current, voltage)

        feature_vector =  np.concatenate((ac_p, ac, dcs, cot, aot, ps))

        return np.asarray(feature_vector)
 
    def fit(self, X, y):

        # Getting unique labels of trainings data
        y = y.tolist()
        classes = list(set([x for x in y if y.count(x) >= 1]))
        classes = sorted(classes)

        for clas in classes:
            r = np.array([]).reshape(0, X[0].size)
            for i,label in enumerate(y):
                if clas == label:

                    # Discard outliners
                    #e = np.linalg.norm(X[i])
                    #if e > MEAN_DICT[clas]-0.5 or e < MEAN_DICT[clas]+0.5:
                    # -----------------

                    r = np.vstack([r, X[i]])
            
            # (19)
            M = r.shape[0]
            psi = np.divide(np.sum(r, axis = 0), M)

            # (20)
            var = np.subtract(r, psi)

            # (21)
            S_t = np.zeros([psi.shape[0], psi.shape[0]])
            for i in range(M):
                S_t +=  np.outer(var[i], var[i])


            eig_vals, eig_vecs = np.linalg.eig(S_t)

            # Evaluate n_components for given accuracy
            index = self.n_components
            if index < 1:
                tot = sum(eig_vals)
                var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
                cum_var_exp = np.cumsum(var_exp)
                index = np.searchsorted(cum_var_exp, index * 100) + 1
                #print ("n_components: " + str(index) + " for class: " + str(clas))

         
            # (23)         
            U_opt = eig_vecs[:index]

            # Update class attributes
            self.singular_values_.update({clas: eig_vals})
            self.explained_variance_ratio_.update({clas: var_exp})
            self.explained_variance_ratio_sum.update({clas: cum_var_exp})

            self.model.update({clas: np.transpose(U_opt)})
            self.var_dict.update({clas: psi})

    def predict(self, X):
        y = []

        for sample in X:
            # Set treshhold
            e = self.tresh_hold
            label = np.nan
            
            for key in self.var_dict:

                # (24)
                psi_e = np.subtract(sample, self.var_dict[key])
                o_e = np.dot(np.transpose(self.model[key]), psi_e)

                # (25)
                psi_e_t = np.dot(self.model[key], o_e)

                # (26)
                e_e = np.linalg.norm(np.subtract(psi_e_t, psi_e))

                if e > e_e:
                    e = e_e
                    label = key

            y.append(label)
        
        return np.asarray(y)

    # Evaluation of AC powers according to the paper
    def __ac_power(self, current, voltage):
        cycle_n = int(current.size / 200) 

        I_e = np.split(current, cycle_n) 
        V_e = np.split(voltage, cycle_n)

        # Using variable names of the paper
        
        # S: Apparent Power, P: Active Power, Q: Reactive Power, D: Distortion Power
        S = []; P = []; Q= []; D = []
    
        # Evaluation of power through (12)–(16) is done every four cycles of the utility frequency with an overlapping of three cycles. 
        for i in range(3, cycle_n, 3):

            tmp_i = np.asarray(I_e[i-3:i+4])
            tmp_i = np.ndarray.flatten(tmp_i)

            tmp_v = np.asarray(V_e[i-3:i+4])
            tmp_v = np.ndarray.flatten(tmp_v)
            
            I_rms = np.sqrt(np.mean(tmp_i**2))
            V_rms = np.sqrt(np.mean(tmp_v**2))

            # S = Irms * Vrms               (13)
            S_tmp = I_rms * V_rms

            # P = Sum(i[n] * v[n]) / N)     (14)
            P_tmp = 0
            for k in range(0,tmp_i.size):
                P_tmp = P_tmp + (tmp_i[k] * tmp_v[k] / tmp_i.size)

            # Q = sqrt(S^2 - P^2)           (15)
            Q_tmp = np.sqrt((S_tmp**2) - (P_tmp**2))

            # D = sqrt(S^2 - Q^2 - P^2)     (12)
            D_tmp = np.sqrt(np.abs((S_tmp**2) - (Q_tmp**2) - (P_tmp**2)))

            S.append(S_tmp)
            P.append(P_tmp)
            Q.append(Q_tmp)
            D.append(D_tmp)
            
        # normalise PQD     (31)
        P = self.__normalise(S, P)
        Q = self.__normalise(S, Q)
        D = self.__normalise(S, D)

        # Addition of the Apparent Power S instead of only using PQD.
        ac_vector = S + P + Q + D
    
        return ac_vector 
    
    def __normalise(self, S, X):
        # (31)
        w = len(S)
        for i in range(w):
            X[i] = np.sign(X[i]) * (X[i]**2 / S[i]**2)
        return X
        

