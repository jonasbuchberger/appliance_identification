import itertools, sys, os
import numpy as np
import matplotlib.pyplot as plt

"""
Plots Confusion Matrix.
Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

Arguments:
- cm (ndarray): Confusion Matrix.
- classes (list): List containing the classes.
Returns:
- None 
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if len(classes)>10:
        plt.figure(figsize=(16,9), dpi= 300)
    else:
        plt.figure(dpi= 300)
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


"""
Some other plotting functions.
"""
def plot_event(x_1, x_2, title):
    plt.figure(figsize=(16,4), dpi = 300)

    y = np.arange(len(x_1))
    plt.suptitle(title)

    plt.subplot(121)
    plt.title(' ')
    #plt.xlabel('Sample')
    plt.ylabel('Current')
    plt.plot(y, x_1, linewidth = 0.7)

    plt.tick_params(
    axis='x',         
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False) 

    plt.subplot(122)
    plt.title(' ')
    #plt.xlabel('Sample')
    plt.ylabel('Voltage')
    plt.plot(y, x_2, linewidth = 0.7)

    plt.tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,        
    labelbottom=False) 

    plt.savefig(os.path.join("reports", "figures", "event_plot.jpeg"))

def plot_dcs(x_1, x_2, title):
    plt.figure(figsize=(16,4), dpi = 300)
    
    x_1 = dcs(x_1)
    x_2 =dcs(x_2)

    y_1 = np.arange(len(x_1))
    y_2 = np.arange(len(x_2))
    plt.suptitle(title + '\n')

    plt.subplot(121)
    plt.title('\n \nDCS Of 1.5 Second Window')
    #plt.xlabel('Sample')
    plt.ylabel('Device Current Signature')
    plt.plot(y_1, x_1, linewidth = 0.7)

    plt.subplot(122)
    plt.title('\n \nDCS Of 0.5 Second Window')
    #plt.xlabel('Sample')
    plt.ylabel('Device Current Signature')
    plt.plot(y_2, x_2, linewidth = 0.7)
    
    #plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "dcs_plot.jpeg"))


def plot_cot(x_1, x_2, title):
    plt.figure(figsize=(16,4), dpi = 300)
    
    x_1 = cot(x_1)
    x_2 =cot(x_2)

    y_1 = np.arange(len(x_1))
    y_2 = np.arange(len(x_2))
    plt.suptitle(title + '\n')

    plt.subplot(121)
    plt.title('\nCOT Of 1.5 Second Window')
    #plt.xlabel('Sample')
    plt.ylabel('Currnet Over Time')
    plt.plot(y_1, x_1, linewidth = 0.7)

    plt.subplot(122)
    plt.title('\nCOT Of 0.5 Second Window')
    #plt.xlabel('Sample')
    plt.ylabel('Currnet Over Time')
    plt.plot(y_2, x_2, linewidth = 0.7)
    
    #plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "cot_plot.jpeg"))

def plot_aot(x_1, x_2, x_11, x_22, title):
    plt.figure(figsize=(16,4), dpi = 300)
    
    x_1 = aot(x_1, x_11)
    x_2 = aot(x_2, x_22)

    y_1 = np.arange(len(x_1))
    y_2 = np.arange(len(x_2))
    plt.suptitle(title + '\n')

    plt.subplot(121)
    plt.title('\nAOT Of 1.5 Second Window')
    #plt.xlabel('Sample')
    plt.ylabel('Admittance Over Time')
    plt.plot(y_1, x_1, linewidth = 0.7)

    plt.subplot(122)
    plt.title('\nAOT Of 0.5 Second Window')
    #plt.xlabel('Sample')
    plt.ylabel('Admittacne Over Time')
    plt.plot(y_2, x_2, linewidth = 0.7)
    
    #plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "aot_plot.jpeg"))






























def dcs(current):
    cycle_n = current.size / 200  # 6000/200

    current_peaks=[]
    current_cycles = np.split(current, cycle_n) 

    for cycle in current_cycles:
        current_peak = np.amax(cycle)
        current_peaks.append(current_peak)

    current_peaks = np.asarray(current_peaks)

    # Split for 0.5 second window
    if current.size == 6000:
        current_peaks_A = np.split(current_peaks, 2)[0]
        current_peaks_B = np.split(current_peaks, 2)[1]

    # Split for 1.5 second windows
    elif current.size == 18000:
        tmp = np.split(current_peaks, 3)
        current_peaks_A = tmp[0]
        current_peaks_B = np.append(tmp[1], tmp[2])
        
    else:
        print("Diffrent window size. Add a rule in my_extra_trees.py")


    # Relative magnitude of the current peaks
    mean_steady_peak_A = np.mean(current_peaks_A)
    device_current_signature = np.subtract(current_peaks_B, mean_steady_peak_A)

    #max = np.max(aot)
    #aot = np.divide(aot, max)
        

    return device_current_signature
    
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

    else:
        print("Diffrent window size. Add a rule in my_extra_trees.py for COT.")
        sys.exit()

    cot = np.asarray(cot)

    return cot

def aot(current, voltage):

    cycle_n = current.size / 200 

    co = cot(current)

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

    else:
        print("Diffrent window size. Add a rule in my_extra_trees.py for AOT.")
        sys.exit()

    aot = np.divide(co, np.asarray(vol))

    return aot