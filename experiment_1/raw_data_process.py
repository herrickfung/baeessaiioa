'''
Input file: /naive_result/raw
Output file: /naive_result/processed
#
The script will read all raw data from the input directory,
and will return a single file for each participants which
will be ready for psychometric curve fitting in fitting.py
'''

import numpy as np
import os
import pandas as pd

input_folder_path =\
    "C:/Users/Herrick Fung/Desktop/Course Materials/Sem 4.1/\
PSY402 Research Thesis II/experiment/naive_result/raw/"
output_folder_path =\
    "C:/Users/Herrick Fung/Desktop/Course Materials/Sem 4.1/\
PSY402 Research Thesis II/experiment/naive_result/processed/"


def remove_two_sd(latency_array, mean, sd):
    adjusted_value = []
    minima = mean - 2 * sd
    maxima = mean + 2 * sd
    for value in latency_array:
        if value < minima or value > maxima:
            pass
        else:
            adjusted_value.append(value)
    return sum(adjusted_value) / len(adjusted_value)


def check_cue_set_coherency(cue_sets):
    bool_cue_ori_array = []
    bool_set_ori_array = []
    coherency_array = []

    cue_oris = cue_sets.Cued_Orientation
    set_oris = cue_sets.Set_Orientation

    for cue_ori in cue_oris:
        if cue_ori == 0:
            bool_cue_ori_array.append(None)
        elif cue_ori > 0:
            bool_cue_ori_array.append(1)
        else:  # cue_ori < 0:
            bool_cue_ori_array.append(0)

    for set_ori in set_oris:
        if set_ori == 0:
            bool_set_ori_array.append(None)
        elif set_ori > 0:
            bool_set_ori_array.append(1)
        else:  # set_ori < 0:
            bool_set_ori_array.append(0)

    for i in range(len(bool_cue_ori_array)):
        if bool_cue_ori_array[i] == None or bool_set_ori_array[i] == None:
            coherency_array.append(None)
        elif (bool_cue_ori_array[i] == bool_set_ori_array[i]):
            coherency_array.append(1)
        else: # unequal
            coherency_array.append(0)

    return coherency_array


def read_and_sort(filename):
    input_filename = f"raw/{filename}"
    output_filename = f"processed/processed_{filename}"

    # Read the input file & return response, j = 1, f = 0,
    input_data = pd.read_csv(input_filename, 'r', delimiter=',')
    input_data['return_resp'] = \
        input_data['Response'].replace(to_replace=["j", "f"], value=[1, 0])
    # Return cue-set coherency, coherent = 1, incoherent = 0, for 0 = None
    input_data['cue_set_coherency'] = \
        check_cue_set_coherency(input_data.iloc[:,[8, 9]])


    # Create array for output data frame
    date_array = []
    time_array = []
    name_array = []
    age_array = []
    gender_array = []
    hand_array = []
    condition_array = []
    neg30_array = []
    neg20_array = []
    neg10_array = []
    zero_array = []
    pos10_array = []
    pos20_array = []
    pos30_array = []
    deg = []
    np.array(deg)
    names = [
        neg30_array, neg20_array, neg10_array,
        zero_array, pos10_array, pos20_array,
        pos30_array
    ]
    for name in names:
        deg.append(name)
    accuracy_10 = []
    accuracy_20 = []
    accuracy_30 = []
    adj_latency_mean_array = []
    coherent_accuracy_10 = []
    coherent_accuracy_20 = []
    coherent_accuracy_30 = []
    incoherent_accuracy_10 = []
    incoherent_accuracy_20 = []
    incoherent_accuracy_30 = []

    # read from input and write from output (Participant Basic info. and Con.)
    date_array = input_data.Exp_Date[0:4]
    time_array = input_data.Exp_Time[0:4]
    name_array = input_data.Sub_Name[0:4]
    age_array = input_data.Age[0:4]
    gender_array = input_data.Gender[0:4]
    hand_array = input_data.Dominant_Hand[0:4]
    condition_array = [1,2,3,4]
    coherency = [1, 0]

    '''
    Obtain mean and n-1 sd for latency for each condition &
    Count no. of "J" response for each orientation and each condition
    '''
    orientation = [-30,-20,-10,0,10,20,30]
    con1_count = []
    con2_count = []
    con3_count = []
    con4_count = []
    condition = []
    np.array(condition)
    for names in [con1_count, con2_count, con3_count, con4_count]:
        condition.append(names)
        np.array(names)

    con1_coherent_array = []
    con2_coherent_array = []
    con3_coherent_array = []
    con4_coherent_array = []
    con1_incoherent_array = []
    con2_incoherent_array = []
    con3_incoherent_array = []
    con4_incoherent_array = []

    cons_and_coherence = []
    np.array(cons_and_coherence)
    for name in [con1_coherent_array, con2_coherent_array, con3_coherent_array, con4_coherent_array, con1_incoherent_array, con2_incoherent_array, con3_incoherent_array, con4_incoherent_array]:
        cons_and_coherence.append(name)
        np.array(name)

    for con in condition_array:
        # for latency
        con_frame = input_data[input_data.Condition == con]
        raw_mean = con_frame['Latency'].mean(axis = 0)
        raw_sd = con_frame['Latency'].std(axis = 0)
        adjusted_mean = remove_two_sd(con_frame['Latency'], raw_mean, raw_sd)
        adj_latency_mean_array.append(adjusted_mean)

        # for response
        for ori in orientation:
            if con == 1 or con == 4:
                ori_frame = con_frame[con_frame.Cued_Orientation == ori]
                if con == 1:
                    con1_count.append(ori_frame['return_resp'].sum(axis = 0))
                    for coh in coherency:
                        if coh == 1:
                            coh_frame = ori_frame[ori_frame.cue_set_coherency == coh]
                            con1_coherent_array.append(coh_frame['return_resp'].sum(axis = 0))
                        else:
                            incoh_frame = ori_frame[ori_frame.cue_set_coherency == coh]
                            con1_incoherent_array.append(incoh_frame['return_resp'].sum(axis = 0))
                else:
                    con4_count.append(ori_frame['return_resp'].sum(axis = 0))
                    for coh in coherency:
                        if coh == 1:
                            coh_frame = ori_frame[ori_frame.cue_set_coherency == coh]
                            con4_coherent_array.append(coh_frame['return_resp'].sum(axis = 0))
                        else:
                            incoh_frame = ori_frame[ori_frame.cue_set_coherency == coh]
                            con4_incoherent_array.append(incoh_frame['return_resp'].sum(axis = 0))
            else:
                ori_frame = con_frame[con_frame.Set_Orientation == ori]
                if con == 2:
                    con2_count.append(ori_frame['return_resp'].sum(axis = 0))
                    for coh in coherency:
                        if coh == 1:
                            coh_frame = ori_frame[ori_frame.cue_set_coherency == coh]
                            con2_coherent_array.append(coh_frame['return_resp'].sum(axis = 0))
                        else:
                            incoh_frame = ori_frame[ori_frame.cue_set_coherency == coh]
                            con2_incoherent_array.append(incoh_frame['return_resp'].sum(axis = 0))
                else:
                    con3_count.append(ori_frame['return_resp'].sum(axis = 0))
                    for coh in coherency:
                        if coh == 1:
                            coh_frame = ori_frame[ori_frame.cue_set_coherency == coh]
                            con3_coherent_array.append(coh_frame['return_resp'].sum(axis = 0))
                        else:
                            incoh_frame = ori_frame[ori_frame.cue_set_coherency == coh]
                            con3_incoherent_array.append(incoh_frame['return_resp'].sum(axis = 0))

    for i in range(4):
        for j in range(7):
            deg[j].append(condition[i][j])
        accuracy_10.append(14 - condition[i][2] + condition[i][4])
        accuracy_20.append(14 - condition[i][1] + condition[i][5])
        accuracy_30.append(14 - condition[i][0] + condition[i][6])

    for i in range(8):
        if i < 4:
            coherent_accuracy_10.append(6 - cons_and_coherence[i][2] + cons_and_coherence[i][4])
            coherent_accuracy_20.append(6 - cons_and_coherence[i][1] + cons_and_coherence[i][5])
            coherent_accuracy_30.append(6 - cons_and_coherence[i][0] + cons_and_coherence[i][6])
        else:
            incoherent_accuracy_10.append(6 - cons_and_coherence[i][2] + cons_and_coherence[i][4])
            incoherent_accuracy_20.append(6 - cons_and_coherence[i][1] + cons_and_coherence[i][5])
            incoherent_accuracy_30.append(6 - cons_and_coherence[i][0] + cons_and_coherence[i][6])

    # Create the output dataframe and saved to output file path
    output_data = pd.DataFrame({'Exp_Date': date_array,
                                'Exp_Time': time_array,
                                'Parti_Name': name_array,
                                'Age': age_array,
                                'Gender': gender_array,
                                'Dominant_Hand': hand_array,
                                'Condition': condition_array,
                                'Count_-30': neg30_array,
                                'Count_-20': neg20_array,
                                'Count_-10': neg10_array,
                                'Count_0': zero_array,
                                'Count_+10': pos10_array,
                                'Count_+20': pos20_array,
                                'Count_+30': pos30_array,
                                'Accuracy_10': accuracy_10,
                                'Accuracy_20': accuracy_20,
                                'Accuracy_30': accuracy_30,
                                'Adj_Latency_Mean': adj_latency_mean_array,
                                'Cohe_Accuracy_10': coherent_accuracy_10,
                                'Cohe_Accuracy_20': coherent_accuracy_20,
                                'Cohe_Accuracy_30': coherent_accuracy_30,
                                'InCohe_Accuracy_10': incoherent_accuracy_10,
                                'InCohe_Accuracy_20': incoherent_accuracy_20,
                                'InCohe_Accuracy_30': incoherent_accuracy_30,
                                })
    output_data.to_csv(output_filename, sep=',', index=False)


def main():
    # Create the output directory
    try:
        os.mkdir(output_folder_path)
        print("Processed Directory Created!")
    except FileExistsError:
        print("Processed Directory Existed!")

    # sort out the main result files from input directory and work on it
    print("Processed the following files:")
    for filename in os.listdir(input_folder_path):
        if filename.endswith("_ep_experiment.csv"):
            print(filename)
            read_and_sort(filename)


if __name__ == "__main__":
    main()
