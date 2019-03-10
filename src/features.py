import sys
import numpy as np
from scipy.signal import correlate

"""
Reference implementation for the following features:
Current Over Time, Admittance Over Time, AC power:  https://dl.acm.org/citation.cfm?id=3077845
By author: Matthias Kahl et al.
Divice Current Signature: https://ieeexplore.ieee.org/document/8284200/
By author: Aashish Kumar Jain et al.
"""


"""
Evaluates Apparent-, Active- and Reactive Power.

Arguments:
- current (Array): Current measurements of event window.
- voltage (Array): Voltage measurements of event window.
Returns:
- ac_vector (list): Contains PQS measurements per cycle.
"""
def ac_power(current, voltage):

    cycle_n = current.size / 200 

    current = np.asarray(current)
    voltage = np.asarray(voltage)

    current_cycles = np.split(current, cycle_n)
    voltage_cycles = np.split(voltage, cycle_n)

    # S: Apparent Power, P: Active Power, Q: Reactive Power, D: Distortion Power
    P = []; Q = []; D = []
    
    # Iterating over all cycles
    for i in range(int(cycle_n)):
            
        I_rms = np.sqrt(np.mean(current_cycles[i]**2))
        V_rms = np.sqrt(np.mean(voltage_cycles[i]**2))

        # S = Irms * Vrms            
        S_tmp = I_rms * V_rms

        # P = Sum(i[n] * v[n]) / N)     
        P_tmp = 0
        for k in range(0,200):
            P_tmp = P_tmp + (current_cycles[i][k] * voltage_cycles[i][k] / 200)

        # Q = sqrt(S^2 - P^2)           
        Q_tmp = np.sqrt((S_tmp**2) - (P_tmp**2))

        # D = sqrt(S^2 - Q^2 - P^2)     
        D_tmp = np.sqrt(np.abs((S_tmp**2) - (Q_tmp**2) - (P_tmp**2)))

        P.append(P_tmp)
        Q.append(Q_tmp)
        D.append(D_tmp)

    ac_vector = P + Q + D

    return ac_vector

"""
Evaluates Device Current Signautre.

Arguments:
- current (Array): Current measurements of event window.
Returns:
- device_current_signature (Array): Contains the DCS.
"""
def dcs(current):

    cycle_n = current.size / 200 

    current_peaks=[]
    current_cycles = np.split(current, cycle_n) 

    for cycle in current_cycles:
        current_peak = np.amax(cycle)
        current_peaks.append(current_peak)

    current_peaks = np.asarray(current_peaks)

    # Split for 0.5 and 1 second window
    if cycle_n == 30 or cycle_n == 60:
        current_peaks_A = np.split(current_peaks, 2)[0]
        current_peaks_B = np.split(current_peaks, 2)[1]

    # Split for 1.5 second windows
    elif cycle_n == 90:
        tmp = np.split(current_peaks, 3)
        current_peaks_A = tmp[0]
        current_peaks_B = np.append(tmp[1], tmp[2])
        
    else:
        print("Diffrent window size. Add a rule in my_extra_trees.py for DCS.")
        sys.exit()


    # Relative magnitude of the current peaks
    mean_steady_peak_A = np.mean(current_peaks_A)

    device_current_signature = np.subtract(current_peaks_B, mean_steady_peak_A)

    return device_current_signature

"""
Evaluates Current Over Time.

Arguments:
- current (Array): Current measurements of event window.
Returns:
- cot (Array): Contains the COT.
"""       
def cot(current):

    cycle_n = current.size / 200 

    cot = []
    current_cycles = np.split(current, cycle_n)  

    for cycle in current_cycles:
        I_p_i = np.sqrt(np.mean(cycle**2))
        cot.append(I_p_i)

    # First 25 cycles for 0.5 second window
    if cycle_n == 30:
        cot = cot[:25]

    # Frist 75 cycles for 1.5 second window
    elif cycle_n == 90:
        cot = cot[:75]

    # First 50 cycles for 1 second window
    elif cycle_n == 60:
        cot = cot[:50]

    else:
        print("Diffrent window size. Add a rule in my_extra_trees.py for COT.")
        sys.exit()

    cot = np.asarray(cot)

    return cot

"""
Evaluates Admittance Over Time.

Arguments:
- current (Array): Current measurements of event window.
- voltage (Array): Voltage measurements of event window.
Returns:
- aot (Array): Contains the AOT.
"""      
def aot(current, voltage):

    cycle_n = current.size / 200 

    c = cot(current)

    vol = []
    voltage_cycles = np.split(voltage, cycle_n) 

    for cycle in voltage_cycles:
        U_p_i = np.sqrt(np.mean(cycle**2))
        vol.append(U_p_i)

    # First 25 cycles for 0.5 second window
    if cycle_n == 30:
        vol = vol[:25]

    # First 75 cycles for 1.5 second window
    elif cycle_n == 90:
        vol = vol[:75]

    # First 50 cycles for 1 second window
    elif cycle_n == 60:
        vol = vol[:50]

    else:
        print("Diffrent window size. Add a rule in my_extra_trees.py for AOT.")
        sys.exit()

    aot = np.divide(c, np.asarray(vol))

    return aot


"""
NOT used for evaluation.
Approximates Phase Shift.
"""
def phase_shift(current, voltage):
    nsamples = current.size

    current -= current.mean(); current /= current.std()
    voltage -= voltage.mean(); voltage /= voltage.std()

    xcorr = correlate(current, voltage)

    dt = np.arange(1-nsamples, nsamples)

    recovered_time_shift = dt[xcorr.argmax()]

    # Time for each sample in ms
    recovered_time_shift = recovered_time_shift * (1/12000) 

    # 2 * pi * f * delta t
    phi_rad = 2 * np.pi * 60 * recovered_time_shift

    return [phi_rad]
