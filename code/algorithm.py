lowanxiety_list = ['test ecg respiration2019-12-13T11_41_08_A109', 'test ecg respiration2020-01-23T11_31_45_A114', 
                    'test ecg respiration2020-01-23T11_31_45_A113', 'test ecg respiration2019-11-15T11_40_16_A105', 'test ecg respiration2020-02-05T14_37_43_A118', 
                    'test ecg respiration2019-11-15T14_43_58_A107', 'test ecg respiration2019-12-13T10_28_27_A108', 'test ecg respiration2020-01-10T11_48_51_A111', 
                    'test ecg respiration2020-01-28T13_16_21_A115', 'test ecg respiration2020-02-26T13_58_51_A121', 'test ecg respiration2019-09-17T12_18_55_A100', 
                    'test ecg respiration2020-02-13T11_03_34_A120', 'test ecg respiration2020-01-10T10_20_07_A110', 'test ecg respiration2020-02-13T11_03_34_A119', 
                    'test ecg respiration2019-09-17T12_18_55_A102', 'test ecg respiration2019-09-17T12_18_55_A103', 'test ecg respiration2019-09-17T12_18_55_A104']
midanxiety_list = ['test ecg respiration2019-11-15T13_24_26_A106', 'test ecg respiration2020-01-28T13_16_21_A116']
anxiety_seg_list = ['seg1.csv','seg3.csv','seg5.csv','seg7.csv']  #,'anxiety_clips.csv'
happy_seg_list = ['seg2.csv','seg4.csv','seg6.csv','seg8.csv']   #,'happy_clips.csv'

from sklearn import svm,neighbors,naive_bayes,discriminant_analysis,neural_network,metrics,tree
from sklearn.cluster import k_means
from sklearn.model_selection import KFold,cross_val_score,RepeatedKFold
import pandas as pd
from function import normalize
import numpy as np
from scipy.stats import sem
from matplotlib import pyplot
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SequentialFeatureSelector,RFE,RFECV
import os

pd.set_option('display.max_rows',None)
'''

# calculate HRV of each segment
for subject in lowanxiety_list:
    segment_path = 'segment1/' + subject
    for segment in anxiety_seg_list:
        anxiety_path = segment_path + '/' + segment
        anxiety_df = pd.read_csv(anxiety_path)
        HRV_path = 'HRV_1min'+'/'+ subject  
        if os.path.exists(HRV_path) == False:
            os.mkdir(HRV_path)
        seg = os.path.splitext(segment)[0]
        hrv_time_anxiety, hrv_freq_anxiety,hrv_non_anxiety,rrv_anxiety = sliding_window(seg= anxiety_df,window = 30000,step = 500) # 
        #hrv_anxiety = pd.concat([hrv_time_anxiety,hrv_freq_anxiety],axis=1)
        hrv_anxiety = pd.concat([hrv_time_anxiety,hrv_freq_anxiety,hrv_non_anxiety,rrv_anxiety],axis=1)
        hrv_anxiety.to_csv(HRV_path +'/' + seg + '_anxiety_features.csv')
        #hrv_time_anxiety.to_csv(HRV_path +'/' + seg + '_anxiety_hrv_time.csv')
        #hrv_freq_anxiety.to_csv(HRV_path +'/' + seg + '_anxiety_hrv_freq.csv')
        #rrv_anxiety.to_csv(HRV_path +'/' + seg + '_anxiety_rrv.csv')

    for segment in happy_seg_list:
        happy_path = segment_path + '/' + segment
        happy_df = pd.read_csv(happy_path)
        HRV_path = 'HRV_1min'+'/'+ subject
        if os.path.exists(HRV_path) == False:
            os.mkdir(HRV_path)
        seg = os.path.splitext(segment)[0]
        hrv_time_happy, hrv_freq_happy,hrv_non_happy,rrv_happy = sliding_window(seg= happy_df,window = 30000,step = 500) # hrv_non_happy,
        #hrv_happy = pd.concat([hrv_time_happy,hrv_freq_happy],axis=1)
        hrv_happy = pd.concat([hrv_time_happy,hrv_freq_happy,hrv_non_happy,rrv_happy],axis=1)
        hrv_happy.to_csv(HRV_path +'/' + seg + '_happy_features.csv')
        #hrv_time_happy.to_csv(HRV_path +'/' + seg + '_happy_hrv_time.csv')
        #hrv_freq_happy.to_csv(HRV_path +'/' + seg + '_happy_hrv_freq.csv')
        #rrv_happy.to_csv(HRV_path +'/' + seg + '_happy_rrv.csv')

for subject in midanxiety_list:
    segment_path = 'segment1/' + subject
    for segment in anxiety_seg_list:
        anxiety_path = segment_path + '/' + segment
        anxiety_df = pd.read_csv(anxiety_path)
        HRV_path = 'HRV_1min'+'/'+ subject  
        if os.path.exists(HRV_path) == False:
            os.mkdir(HRV_path)
        seg = os.path.splitext(segment)[0]
        hrv_time_anxiety, hrv_freq_anxiety,hrv_non_anxiety,rrv_anxiety = sliding_window(seg= anxiety_df,window = 30000,step = 500) # 
        #hrv_anxiety = pd.concat([hrv_time_anxiety,hrv_freq_anxiety],axis=1)
        hrv_anxiety = pd.concat([hrv_time_anxiety,hrv_freq_anxiety,hrv_non_anxiety,rrv_anxiety],axis=1)
        hrv_anxiety.to_csv(HRV_path +'/' + seg + '_anxiety_features.csv')
        #hrv_time_anxiety.to_csv(HRV_path +'/' + seg + '_anxiety_hrv_time.csv')
        #hrv_freq_anxiety.to_csv(HRV_path +'/' + seg + '_anxiety_hrv_freq.csv')
        #rrv_anxiety.to_csv(HRV_path +'/' + seg + '_anxiety_rrv.csv')

    for segment in happy_seg_list:
        happy_path = segment_path + '/' + segment
        happy_df = pd.read_csv(happy_path)
        HRV_path = 'HRV_1min'+'/'+ subject
        if os.path.exists(HRV_path) == False:
            os.mkdir(HRV_path)
        seg = os.path.splitext(segment)[0]
        hrv_time_happy, hrv_freq_happy,hrv_non_happy,rrv_happy = sliding_window(seg= happy_df,window = 30000,step = 500) # hrv_non_happy,
        #hrv_happy = pd.concat([hrv_time_happy,hrv_freq_happy],axis=1)
        hrv_happy = pd.concat([hrv_time_happy,hrv_freq_happy,hrv_non_happy,rrv_happy],axis=1)
        hrv_happy.to_csv(HRV_path +'/' + seg + '_happy_features.csv')
        #hrv_time_happy.to_csv(HRV_path +'/' + seg + '_happy_hrv_time.csv')
        #hrv_freq_happy.to_csv(HRV_path +'/' + seg + '_happy_hrv_freq.csv')
        #rrv_happy.to_csv(HRV_path +'/' + seg + '_happy_rrv.csv')


print('finish')
print('-------------')
'''
anxiety_mid = pd.DataFrame()
anxiety_mid2 = pd.DataFrame()
happy_mid = pd.DataFrame()
happy_mid2 = pd.DataFrame()
anxiety_low = pd.DataFrame()
happy_low = pd.DataFrame()
anxiety_low2 = pd.DataFrame()
happy_low2 = pd.DataFrame()

'''
# connect all anxiety data and happy data together for all subjects
for subject in midanxiety_list:
    anxiety_path1 = 'HRV_1min/'+ subject +'/seg1_anxiety_features.csv'
    anxiety_path2 = 'HRV_1min/'+ subject +'/seg3_anxiety_features.csv'
    anxiety_path3 = 'HRV_1min/'+ subject +'/seg5_anxiety_features.csv'
    anxiety_path4 = 'HRV_1min/'+ subject +'/seg7_anxiety_features.csv'

    # anxiety_rrv1 = 'hrv2/'+ subject +'/seg1_anxiety_rrv.csv'
    # anxiety_rrv2 = 'hrv2/'+ subject +'/seg3_anxiety_rrv.csv'
    # anxiety_rrv3 = 'hrv2/'+ subject +'/seg5_anxiety_rrv.csv'
    # anxiety_rrv4 = 'hrv2/'+ subject +'/seg7_anxiety_rrv.csv'

    happy_path1 = 'HRV_1min/'+ subject +'/seg2_happy_features.csv'
    happy_path2 = 'HRV_1min/'+ subject +'/seg4_happy_features.csv'
    happy_path3 = 'HRV_1min/'+ subject +'/seg6_happy_features.csv'
    happy_path4 = 'HRV_1min/'+ subject +'/seg8_happy_features.csv'

    # happy_rrv1 = 'hrv2/'+ subject +'/seg2_happy_rrv.csv'
    # happy_rrv2 = 'hrv2/'+ subject +'/seg4_happy_rrv.csv'
    # happy_rrv3 = 'hrv2/'+ subject +'/seg6_happy_rrv.csv'
    # happy_rrv4 = 'hrv2/'+ subject +'/seg8_happy_rrv.csv'

    anxiety_df1 = pd.read_csv(anxiety_path1)
    anxiety_df2 = pd.read_csv(anxiety_path2)
    anxiety_df3 = pd.read_csv(anxiety_path3)
    anxiety_df4 = pd.read_csv(anxiety_path4)
    # anxiety_rrv_df1 = pd.read_csv(anxiety_rrv1)
    # anxiety_rrv_df2 = pd.read_csv(anxiety_rrv2)
    # anxiety_rrv_df3 = pd.read_csv(anxiety_rrv3)
    # anxiety_rrv_df4 = pd.read_csv(anxiety_rrv4)

    happy_df1 = pd.read_csv(happy_path1)
    happy_df2 = pd.read_csv(happy_path2)
    happy_df3 = pd.read_csv(happy_path3)
    happy_df4 = pd.read_csv(happy_path4)
    # happy_rrv_df1 = pd.read_csv(happy_rrv1)
    # happy_rrv_df2 = pd.read_csv(happy_rrv2)
    # happy_rrv_df3 = pd.read_csv(happy_rrv3)
    # happy_rrv_df4 = pd.read_csv(happy_rrv4)

    anxiety_df = pd.concat([anxiety_df1,anxiety_df2,anxiety_df3,anxiety_df4],axis=0)
    happy_df = pd.concat([happy_df1,happy_df2,happy_df3,happy_df4],axis=0)
    anxiety_mid = pd.concat([anxiety_df,anxiety_mid],axis=0)
    happy_mid = pd.concat([happy_df,happy_mid],axis=0)
    data_path = 'HRV_1min/classification'
    anxiety_mid.to_csv(data_path+'/anxiety_hrv_low.csv')
    happy_mid.to_csv(data_path+'/happy_hrv_low.csv')

    # anxiety_rrv_df = pd.concat([anxiety_rrv_df1,anxiety_rrv_df2,anxiety_rrv_df3,anxiety_rrv_df4],axis=0)
    # anxiety_hrv_rrv = pd.concat([anxiety_df,anxiety_rrv_df],axis=1)
    # print(subject)
    # #anxiety_hrv_rrv= anxiety_hrv_rrv.loc[~anxiety_hrv_rrv.index.duplicated(keep='first')]
    # anxiety_mid2 = pd.concat([anxiety_hrv_rrv,anxiety_mid2],axis=0)
    # anxiety_mid2.to_csv(data_path+'/anxiety_hrv_rrv_mid.csv')

    # happy_rrv_df = pd.concat([happy_rrv_df1,happy_rrv_df2,happy_rrv_df3,happy_rrv_df4],axis=0)
    # happy_hrv_rrv = pd.concat([happy_df,happy_rrv_df],axis=1)
    # happy_mid2 = pd.concat([happy_hrv_rrv,happy_mid2],axis=0)
    # happy_mid2.to_csv(data_path+'/happy_hrv_rrv_mid.csv')



for subject in lowanxiety_list:
    anxiety_path1 = 'hrv2/'+ subject +'/seg1_anxiety_hrv.csv'
    anxiety_path2 = 'hrv2/'+ subject +'/seg3_anxiety_hrv.csv'
    anxiety_path3 = 'hrv2/'+ subject +'/seg5_anxiety_hrv.csv'
    anxiety_path4 = 'hrv2/'+ subject +'/seg7_anxiety_hrv.csv'

    anxiety_rrv1 = 'hrv2/'+ subject +'/seg1_anxiety_rrv.csv'
    anxiety_rrv2 = 'hrv2/'+ subject +'/seg3_anxiety_rrv.csv'
    anxiety_rrv3 = 'hrv2/'+ subject +'/seg5_anxiety_rrv.csv'
    anxiety_rrv4 = 'hrv2/'+ subject +'/seg7_anxiety_rrv.csv'

    happy_path1 = 'hrv2/'+ subject +'/seg2_happy_hrv.csv'
    happy_path2 = 'hrv2/'+ subject +'/seg4_happy_hrv.csv'
    happy_path3 = 'hrv2/'+ subject +'/seg6_happy_hrv.csv'
    happy_path4 = 'hrv2/'+ subject +'/seg8_happy_hrv.csv'

    happy_rrv1 = 'hrv2/'+ subject +'/seg2_happy_rrv.csv'
    happy_rrv2 = 'hrv2/'+ subject +'/seg4_happy_rrv.csv'
    happy_rrv3 = 'hrv2/'+ subject +'/seg6_happy_rrv.csv'
    happy_rrv4 = 'hrv2/'+ subject +'/seg8_happy_rrv.csv'

    anxiety_df1 = pd.read_csv(anxiety_path1)
    anxiety_df2 = pd.read_csv(anxiety_path2)
    anxiety_df3 = pd.read_csv(anxiety_path3)
    anxiety_df4 = pd.read_csv(anxiety_path4)

    anxiety_rrv_df1 = pd.read_csv(anxiety_rrv1)
    anxiety_rrv_df2 = pd.read_csv(anxiety_rrv2)
    anxiety_rrv_df3 = pd.read_csv(anxiety_rrv3)
    anxiety_rrv_df4 = pd.read_csv(anxiety_rrv4)

    happy_df1 = pd.read_csv(happy_path1)
    happy_df2 = pd.read_csv(happy_path2)
    happy_df3 = pd.read_csv(happy_path3)
    happy_df4 = pd.read_csv(happy_path4)

    happy_rrv_df1 = pd.read_csv(happy_rrv1)
    happy_rrv_df2 = pd.read_csv(happy_rrv2)
    happy_rrv_df3 = pd.read_csv(happy_rrv3)
    happy_rrv_df4 = pd.read_csv(happy_rrv4)

    anxiety_df = pd.concat([anxiety_df1,anxiety_df2,anxiety_df3,anxiety_df4],axis=0)
    happy_df = pd.concat([happy_df1,happy_df2,happy_df3,happy_df4],axis=0)
    anxiety_low = pd.concat([anxiety_df,anxiety_low],axis=0)
    happy_low = pd.concat([happy_df,happy_low],axis=0)

    data_path = 'hrv2/classification_data'
    # anxiety_low.to_csv(data_path+'/anxiety_hrv_low.csv')
    # happy_low.to_csv(data_path+'/happy_hrv_low.csv')

    anxiety_rrv_df = pd.concat([anxiety_rrv_df1,anxiety_rrv_df2,anxiety_rrv_df3,anxiety_rrv_df4],axis=0)
    anxiety_rrv_df = anxiety_rrv_df[['RRV_RMSSD', 'RRV_MeanBB', 'RRV_SDBB', 'RRV_SDSD', 'RRV_VLF', 'RRV_LF', 'RRV_HF', 'RRV_LFHF', 'RRV_SD1', 'RRV_SD2', 'RRV_SD2SD1']]
    anxiety_hrv_rrv = pd.concat([anxiety_df,anxiety_rrv_df],axis=1)
    # print(anxiety_hrv_rrv.columns)
    print(subject)
    # #anxiety_hrv_rrv= anxiety_hrv_rrv.loc[~anxiety_hrv_rrv.index.duplicated(keep='first')]
    anxiety_low2 = pd.concat([anxiety_hrv_rrv,anxiety_low2],axis=0)
    # anxiety_low2.to_csv(data_path+'/anxiety_hrv_rrv_low.csv')

    happy_rrv_df = pd.concat([happy_rrv_df1,happy_rrv_df2,happy_rrv_df3,happy_rrv_df4],axis=0)
    happy_rrv_df = happy_rrv_df[['RRV_RMSSD', 'RRV_MeanBB', 'RRV_SDBB', 'RRV_SDSD', 'RRV_VLF', 'RRV_LF', 'RRV_HF', 'RRV_LFHF', 'RRV_SD1', 'RRV_SD2', 'RRV_SD2SD1']]
    happy_hrv_rrv = pd.concat([happy_df,happy_rrv_df],axis=1)
    happy_low2 = pd.concat([happy_hrv_rrv,happy_low2],axis=0)
    # happy_low2.to_csv(data_path+'/happy_hrv_rrv_low.csv')

print(happy_low.shape)
print(happy_low2.shape)
anxiety_low2.to_csv(data_path+'/anxiety_hrv_rrv_low.csv')
happy_low2.to_csv(data_path+'/happy_hrv_rrv_low.csv')
'''


anxiety_mid1 = pd.read_csv('HRV_1min/classification/anxiety_hrv_mid.csv') 
happy_mid1 = pd.read_csv('HRV_1min/classification/happy_hrv_mid.csv')
anxiety_low1 = pd.read_csv('HRV_1min/classification/anxiety_hrv_low.csv')
happy_low1 = pd.read_csv('HRV_1min/classification/happy_hrv_low.csv')
anxiety_sample = anxiety_low1.shape[0] + anxiety_mid1.shape[0]
happy_sample = happy_low1.shape[0] + happy_mid1.shape[0]
print('Anxiety sample amount: ', anxiety_sample)
print('happy sample amount: ', happy_sample)
print('----------------------------')



# use "1" to label anxiety data
anxiety_data = pd.concat([anxiety_low1,anxiety_mid1],axis=0)
label_anxiety = np.ones(anxiety_data.shape[0])
anxiety_data['label'] = label_anxiety
# anxiety_data.to_csv('anxiety1min.csv')
# print(anxiety_data)

# use "0" to label non-anxiety data
happy_data = pd.concat([happy_low1,happy_mid1],axis=0)
label_happy = np.zeros(happy_data.shape[0])
happy_data['label'] = label_happy
happy_data.to_csv('happy1min.csv')
# print(anxiety_data.shape)
# print(happy_data.shape)

# connect anxiety and non-anxity data together
alldata = pd.concat([anxiety_data,happy_data],axis=0,ignore_index=True)
# drop NaN
alldata = alldata.apply(pd.to_numeric, errors='coerce')
alldata.dropna(how='all', axis=1, inplace=True)
alldata.drop(alldata.columns[[0,1,92,91,90,89,88,87,86,85,84,83,82,81]], axis=1, inplace=True)
alldata.drop(alldata.columns[24:61], axis=1, inplace=True)
print(alldata.columns)
alldata.to_csv('HRV_1min/classification/twoleveldata_hrvrrv.csv')
print(alldata.shape)


alldata_norm = normalize(alldata)