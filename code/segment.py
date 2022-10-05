import pandas as pd
import scipy.io
import os


data_path = ('Data')
filelist = os.listdir(data_path)
file_list =[]
#ingore the hidden file 
for file in filelist:
    if not file.startswith('.') and os.path.isfile(os.path.join(data_path, file)):
        file_list.append(file)


# do a loop to process the data of each subject
for file in file_list:
    # read data from .mat file
    file_name = os.path.basename(data_path + '/'+file)
    folder_name = os.path.splitext(file_name)[0]
    data = scipy.io.loadmat(data_path + '/'+file)
    con_list = [[element for element in upperElement] for upperElement in data['data']]
    columns = ['ecg', 'rsp']
    df = pd.DataFrame(con_list, columns=columns)

    # segment the data based on video clips
    df_seg1 = df[2506:100505]
    df_seg2 = df[100506:160005]
    df_seg3 = df[160006:269005]
    df_seg4 = df[269006:317505]
    df_seg5 = df[317506:476005]
    df_seg6 = df[476006:591505]
    df_seg7 = df[591506:1032505]
    df_seg8 = df[1032506:1242505]

    # combine all anxiety/happy clips together
    df_anxiety = pd.concat(([df_seg1,df_seg3,df_seg5,df_seg7]), ignore_index=True)
    df_happy = pd.concat(([df_seg2,df_seg4,df_seg6,df_seg8]), ignore_index=True)

    # segment path
    segment_path = 'segment/' + folder_name
    if os.path.exists(segment_path) == False:
        os.mkdir(segment_path)

    #generate .csv file of segments
    df_seg1.to_csv(segment_path +'/seg1.csv')
    df_seg2.to_csv(segment_path +'/seg2.csv')
    df_seg3.to_csv(segment_path +'/seg3.csv')
    df_seg4.to_csv(segment_path +'/seg4.csv')
    df_seg5.to_csv(segment_path +'/seg5.csv')
    df_seg6.to_csv(segment_path +'/seg6.csv')
    df_seg7.to_csv(segment_path +'/seg7.csv')
    df_seg8.to_csv(segment_path +'/seg8.csv')
    df_anxiety.to_csv(segment_path +'/anxiety_clips.csv')
    df_happy.to_csv(segment_path +'/happy_clips.csv')

print('finish')