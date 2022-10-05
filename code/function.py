from locale import normalize
import scipy.io
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pyphysio as ph
import neurokit2 as nk
from sklearn import svm,neighbors,naive_bayes,discriminant_analysis,neural_network,metrics,tree
from sklearn.model_selection import KFold,cross_val_score
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import sem
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# function to calculate HRV
def get_HRV(df_seg):
    fsamp = 500
    #tstart_ecg = 0
    #ecg = ph.EvenlySignal(values=df_seg['ecg'],sampling_freq= fsamp,signal_type='ecg',start_time=tstart_ecg)
    ecg = df_seg['ecg']
    # butterworth bandpass: 8-20Hz
    ecg_filtered = nk.ecg_clean(ecg, sampling_rate= fsamp,method='elgendi2010') # 
    # Find peaks
    peaks, info = nk.ecg_peaks(ecg_filtered, sampling_rate= fsamp) #,method='elgendi2010'
    #time domain
    hrv_time = nk.hrv_time(peaks, sampling_rate=fsamp, show=False)
    #frequency domian
    hrv_freq = nk.hrv_frequency(peaks, sampling_rate=fsamp, show=False, normalize=False)
    #nonlinear
    hrv_non = nk.hrv_nonlinear(peaks, sampling_rate=fsamp, show=False)
    return hrv_time,hrv_freq,hrv_non

# function to calculate RRV
def get_RRV(df_seg):
    fsamp = 2048
    tstart_rsp = 0
    rsp = ph.EvenlySignal(values=df_seg['rsp'],sampling_freq= fsamp,signal_type='rsp',start_time=tstart_rsp)
    #rsp_normalization = ph.Normalize(norm_method='standard')(rsp)

    plt.rcParams['figure.figsize'] = [10, 6]  # Bigger images
    cleaned = nk.rsp_clean(rsp, sampling_rate=50)

    # Extract peaks
    df, peaks_dict = nk.rsp_peaks(cleaned)
    info = nk.rsp_fixpeaks(peaks_dict)
    formatted = nk.signal_formatpeaks(info, desired_length=len(cleaned),peak_indices=info["RSP_Peaks"])
    #nk.signal_plot(pd.DataFrame({"RSP_Raw": rsp, "RSP_Clean": cleaned}), sampling_rate=50, subplots=True)
    #candidate_peaks = nk.events_plot(peaks_dict['RSP_Peaks'], cleaned)
    #fixed_peaks = nk.events_plot(info['RSP_Peaks'], cleaned)

    # Extract rate
    rsp_rate = nk.rsp_rate(cleaned, peaks_dict, sampling_rate= 50)
    '''
    # Visualize
    plt.figure()
    plt.plot(rsp_rate,label='BPM')
    plt.legend()
    plt.show()
    '''
    rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=50, show=False)
    return rrv


# sliding window function: output-> 1 min time window, 1 second overlapping step 
def sliding_window(seg, window, step):

    start = 5
    end = len(seg)
    n=0
    window_end = 0
    hrv_t = pd.DataFrame()
    hrv_f = pd.DataFrame()
    hrv_n = pd.DataFrame()
    rrv = pd.DataFrame()
    while window_end < end: 
        window_start = start+ (n * step)
        window_end = start + window + (n * step)
        seg_slide = seg.iloc[window_start: window_end]
        hrv_time, hrv_freq,hrv_non = get_HRV(seg_slide)  #,hrv_non
        rrv_seg = get_RRV(seg_slide)
        hrv_t = pd.concat(([hrv_t,hrv_time]), ignore_index=True)
        hrv_f = pd.concat(([hrv_f,hrv_freq]), ignore_index=True)
        hrv_n = pd.concat(([hrv_n,hrv_non]), ignore_index=True)
        #print(hrv_t)
        rrv = pd.concat(([rrv,rrv_seg]), ignore_index=True)
        n += 1
    return hrv_t,hrv_f,hrv_n,rrv


# Max-Min normalization function
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# function to generate cofusion plot
def confusion_plot(actual_classes,predicted_classes):
    cm = confusion_matrix(actual_classes, predicted_classes, normalize='true')
    ax = plt.subplot()
    sns.set(font_scale=1.5) # Adjust to fit
    sns.heatmap(cm, annot=True, ax=ax, cmap="Blues",square=True);  

    # Labels, title and ticks
    label_font = {'size':'15'}  # Adjust to fit
    ax.set_xlabel('Predicted labels', fontdict=label_font);
    ax.set_ylabel('True labels', fontdict=label_font);

    title_font = {'size':'22'}  # Adjust to fit
    ax.set_title('Confusion Matrix', fontdict=title_font);

    ax.tick_params(axis='both', which='major', labelsize=15)  # Adjust to fit
    ax.xaxis.set_ticklabels(['Without Anxiety', 'With Anxiety']);
    ax.yaxis.set_ticklabels(['Without Anxiety', 'With Anxiety']);
    plt.show()


#function to genrate the accuracy of each iteration
def train_classifier(inputdata,n_splits,classifier,classifier_name):

    kf = KFold(n_splits=n_splits,shuffle=True)
    k=0
    accuracy = 0
    acc_list = []
    cm_list = []
    I0 = pd.DataFrame()
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    
    print("-------------------------------------------------")
    print("The result of ",classifier_name)
    for train_index, test_index in kf.split(inputdata):  
        X_train, X_test = inputdata.iloc[train_index], inputdata.iloc[test_index]
        train_samples = len(X_train)
        test_samples = len(X_test)
        # input_train = X_train.iloc[:, 0:42]
        # input_test = X_test.iloc[:, 0:42]

        input_train = X_train.iloc[:,:-1]
        input_test = X_test.iloc[:,:-1]
        classfier_train = classifier
        classfier_train.fit(input_train,X_train['label'])
        y_pred = classfier_train.predict(input_test)
        accuracy_test = metrics.accuracy_score(X_test['label'], y_pred)

        actual_classes = np.append(actual_classes, X_test['label'])
        predicted_classes = np.append(predicted_classes, y_pred)
        k += 1
        print("Accuracy of fold k=", str(k),': ', accuracy_test)


        cm = confusion_matrix(X_test['label'], y_pred, normalize='true')
        acc_list.append(accuracy_test)
        cm_list.append(cm)
        accuracy += accuracy_test

    print('we have ',train_samples,' train samples')
    print('we have ',test_samples,' test samples')
    print("The average accuracy is: ", accuracy/n_splits)
    # print(I0)
    confusion_plot(actual_classes,predicted_classes)
    print(' ')
    return acc_list,cm_list