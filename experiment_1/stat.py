'''
Input file: /naive_result/fitted_data
Output file: /part1_result/merged.csv; /part1_result/stats.csv
#
The script will read all the data from the fitted_data file and merged all to a
single csv file (merged.csv), then, conduct statistical analysis on the merged.csv
file, statistical analysis output will be presented in the stats.csv file
'''

from matplotlib import style
from scipy.stats import ttest_rel, ttest_1samp
from statsmodels.stats.anova import AnovaRM
import matplotlib.font_manager as font_manager
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import pingouin

# input_folder_path = \
#     "C:/Users/Herrick Fung/Desktop/Course Materials/Sem 4.1/\
# PSY402 Research Thesis II/experiment/naive_result/fitted_data/"
# output_merged_file_name = \
#     "C:/Users/Herrick Fung/Desktop/Course Materials/Sem 4.1/\
# PSY402 Research Thesis II/part1_result/merged.csv"
# output_stats_file_name = \
#     "C:/Users/Herrick Fung/Desktop/Course Materials/Sem 4.1/\
# PSY402 Research Thesis II/part1_result/stats.txt"


# input_folder_path = \
#     "~/Desktop/Course Materials/Sem 4.1/\
# PSY402 Research Thesis II/experiment/naive_result/fitted_data/"
input_folder_path = 'fitted_data'
output_merged_file_name = \
    "~/Desktop/Course Materials/Sem 4.1/\
PSY402 Research Thesis II/part1_result/merged.csv"
output_stats_file_name = "stats.txt"
    # "~/Desktop/Course Materials/Sem 4.1/\
# PSY402 Research Thesis II/part1_result/stats.txt"

def read_merge_output():
    merged_array = []
    for filename in os.listdir(input_folder_path):
        input_filename = f"fitted_data/{filename}"
        # input_filename = os.join(fitted_data, filename)
        input_data = pd.read_csv(input_filename, delimiter=',')
        transformed = np.array([input_data.Exp_Date[0],
                                input_data.Exp_Time[0],
                                input_data.Parti_Name[0],
                                input_data.Age[0],
                                input_data.Gender[0],
                                input_data.Dominant_Hand[0],
                                ])
        for i in range(4):
            transformed = np.append(transformed, input_data.R_Squared[i])
            transformed = np.append(transformed, input_data.Alpha[i])
            transformed = np.append(transformed, input_data.Beta[i])
            transformed = np.append(transformed, input_data.Total_Accuracy[i])
            transformed = np.append(transformed, input_data.Cohe_Total_Acc[i])
            transformed = np.append(transformed, input_data.InCohe_Total_Acc[i])
            transformed = np.append(transformed, input_data.Adj_Latency_Mean[i])
            transformed = np.append(transformed, input_data.Accuracy_10[i])
            transformed = np.append(transformed, input_data.Accuracy_20[i])
            transformed = np.append(transformed, input_data.Accuracy_30[i])
            transformed = np.append(transformed, input_data.Cohe_10_Acc[i])
            transformed = np.append(transformed, input_data.Cohe_20_Acc[i])
            transformed = np.append(transformed, input_data.Cohe_30_Acc[i])
            transformed = np.append(transformed, input_data.InCohe_10_Acc[i])
            transformed = np.append(transformed, input_data.InCohe_20_Acc[i])
            transformed = np.append(transformed, input_data.InCohe_30_Acc[i])

        merged_array.append(transformed)

    output_data = pd.DataFrame(merged_array)
    output_data.columns = [

        # Basic Info
                           "Exp_Date",
                           "Exp_Time",
                           "Parti_Name",
                           "Age",
                           "Gender",
                           "Dominant_Hand",

        # Condition 1 Info
                           "C1_R_Squared",
                           "C1_Alpha",
                           "C1_Beta",
                           "C1_Total_Acc",
                           "C1_Cohe_Total_Acc",
                           "C1_InCohe_Total_Acc",
                           "C1_Adj_Mean_Lat",
                           "C1_10_Acc",
                           "C1_20_Acc",
                           "C1_30_Acc",
                           "C1_Cohe_10_Acc",
                           "C1_Cohe_20_Acc",
                           "C1_Cohe_30_Acc",
                           "C1_InCohe_10_Acc",
                           "C1_InCohe_20_Acc",
                           "C1_InCohe_30_Acc",

        # Condition 2 Info
                           "C2_R_Squared",
                           "C2_Alpha",
                           "C2_Beta",
                           "C2_Total_Acc",
                           "C2_Cohe_Total_Acc",
                           "C2_InCohe_Total_Acc",
                           "C2_Adj_Mean_Lat",
                           "C2_10_Acc",
                           "C2_20_Acc",
                           "C2_30_Acc",
                           "C2_Cohe_10_Acc",
                           "C2_Cohe_20_Acc",
                           "C2_Cohe_30_Acc",
                           "C2_InCohe_10_Acc",
                           "C2_InCohe_20_Acc",
                           "C2_InCohe_30_Acc",

        # Condition 3 Info
                           "C3_R_Squared",
                           "C3_Alpha",
                           "C3_Beta",
                           "C3_Total_Acc",
                           "C3_Cohe_Total_Acc",
                           "C3_InCohe_Total_Acc",
                           "C3_Adj_Mean_Lat",
                           "C3_10_Acc",
                           "C3_20_Acc",
                           "C3_30_Acc",
                           "C3_Cohe_10_Acc",
                           "C3_Cohe_20_Acc",
                           "C3_Cohe_30_Acc",
                           "C3_InCohe_10_Acc",
                           "C3_InCohe_20_Acc",
                           "C3_InCohe_30_Acc",

        # Condition 4 Info
                           "C4_R_Squared",
                           "C4_Alpha",
                           "C4_Beta",
                           "C4_Total_Acc",
                           "C4_Cohe_Total_Acc",
                           "C4_InCohe_Total_Acc",
                           "C4_Adj_Mean_Lat",
                           "C4_10_Acc",
                           "C4_20_Acc",
                           "C4_30_Acc",
                           "C4_Cohe_10_Acc",
                           "C4_Cohe_20_Acc",
                           "C4_Cohe_30_Acc",
                           "C4_InCohe_10_Acc",
                           "C4_InCohe_20_Acc",
                           "C4_InCohe_30_Acc",
                           ]

    output_data.to_csv(output_merged_file_name, sep=',', index=False)
    return output_data


def paired_t(a, b):
    print("\n******************************************************************")
    print(f"Paired-T-test on {a.name} & {b.name}")
    print("********************************************************************")
    print(f"Variable is {a.name};  Mean is {a.mean()}; SD is {a.std()}")
    print(f"Variable is {b.name};  Mean is {b.mean()}; SD is {b.std()}")
    diff = a - b
    print(f"Variable is {a.name} - {b.name};  Mean is {diff.mean()}; SD is {diff.std()}")
    r = np.corrcoef(a, b)
    r = r[0,1]
    print(pingouin.ttest(a,b,paired=True))


def one_sample_chance(var, chance):
    print("\n******************************************************************")
    print(f"One-Sample-T-test on {var.name} & Chance at {chance}")
    print("********************************************************************")
    print(f"Variable is {var.name}; Mean is {var.mean()}; SD is {var.std()}")
    diff = var.mean() - chance
    print(f"Chance Difference is {diff}")
    print(pingouin.ttest(var,chance))


def analysis(data):
    print("\n####################################################################")
    print("Cross Conditions, Total Accuracy & Latency Comparison")
    print("####################################################################")
    print("\n####################################################################")
    print("Accuracy")
    print("####################################################################")
    paired_t(data["C1_Total_Acc"].astype(float)/84*100, data["C4_Total_Acc"].astype(float)/84*100)
    paired_t(data["C2_Total_Acc"].astype(float)/84*100, data["C3_Total_Acc"].astype(float)/84*100)
    paired_t(data["C1_Total_Acc"].astype(float)/84*100, data["C3_Total_Acc"].astype(float)/84*100)
    paired_t(data["C2_Total_Acc"].astype(float)/84*100, data["C4_Total_Acc"].astype(float)/84*100)
    print("Pre-Cue")
    paired_t(data["C1_Total_Acc"].astype(float)/84*50 + data["C2_Total_Acc"].astype(float)/84*50, data["C3_Total_Acc"].astype(float)/84*50 + data["C4_Total_Acc"].astype(float)/84*50)
    print("Post_Cue")
    paired_t(data["C1_Total_Acc"].astype(float)/84*50 + data["C4_Total_Acc"].astype(float)/84*50, data["C3_Total_Acc"].astype(float)/84*50 + data["C2_Total_Acc"].astype(float)/84*50)
    print("\n####################################################################")
    print("Latency")
    print("####################################################################")
    paired_t(data["C1_Adj_Mean_Lat"].astype(float)*1000, data["C4_Adj_Mean_Lat"].astype(float)*1000)
    paired_t(data["C2_Adj_Mean_Lat"].astype(float)*1000, data["C3_Adj_Mean_Lat"].astype(float)*1000)
    paired_t(data["C1_Adj_Mean_Lat"].astype(float)*1000, data["C3_Adj_Mean_Lat"].astype(float)*1000)
    paired_t(data["C2_Adj_Mean_Lat"].astype(float)*1000, data["C4_Adj_Mean_Lat"].astype(float)*1000)
    print("Pre-Cue")
    paired_t(data["C1_Adj_Mean_Lat"].astype(float)*500 + data["C2_Adj_Mean_Lat"].astype(float)*500, data["C3_Adj_Mean_Lat"].astype(float)*500 + data["C4_Adj_Mean_Lat"].astype(float)*500)
    print("Post_Cue")
    paired_t(data["C1_Adj_Mean_Lat"].astype(float)*500 + data["C4_Adj_Mean_Lat"].astype(float)*500, data["C3_Adj_Mean_Lat"].astype(float)*500 + data["C2_Adj_Mean_Lat"].astype(float)*500)
    print("\n####################################################################")
    print("R_squared")
    print("####################################################################")
    paired_t(data["C1_R_Squared"].astype(float), data["C4_R_Squared"].astype(float))
    paired_t(data["C2_R_Squared"].astype(float), data["C3_R_Squared"].astype(float))
    paired_t(data["C1_R_Squared"].astype(float), data["C3_R_Squared"].astype(float))
    paired_t(data["C2_R_Squared"].astype(float), data["C4_R_Squared"].astype(float))

    print("\n####################################################################")
    print("Same Condition, Set Cue Coherent VS Set Cue Incoherent")
    print("####################################################################")
    paired_t(data["C1_Cohe_Total_Acc"].astype(float)/36*100, data["C1_InCohe_Total_Acc"].astype(float)/36*100)
    paired_t(data["C2_Cohe_Total_Acc"].astype(float)/36*100, data["C2_InCohe_Total_Acc"].astype(float)/36*100)
    paired_t(data["C3_Cohe_Total_Acc"].astype(float)/36*100, data["C3_InCohe_Total_Acc"].astype(float)/36*100)
    paired_t(data["C4_Cohe_Total_Acc"].astype(float)/36*100, data["C4_InCohe_Total_Acc"].astype(float)/36*100)

    print("\n####################################################################")
    print("Set Cue Coherent Accuracy, compared to Chance Level at 18")
    print("####################################################################")
    one_sample_chance(data["C1_Cohe_Total_Acc"].astype(float)/36*100, 50)
    one_sample_chance(data["C1_InCohe_Total_Acc"].astype(float)/36*100, 50)
    one_sample_chance(data["C2_Cohe_Total_Acc"].astype(float)/36*100, 50)
    one_sample_chance(data["C2_InCohe_Total_Acc"].astype(float)/36*100, 50)
    one_sample_chance(data["C3_Cohe_Total_Acc"].astype(float)/36*100, 50)
    one_sample_chance(data["C3_InCohe_Total_Acc"].astype(float)/36*100, 50)
    one_sample_chance(data["C4_Cohe_Total_Acc"].astype(float)/36*100, 50)
    one_sample_chance(data["C4_InCohe_Total_Acc"].astype(float)/36*100, 50)

    print("\n####################################################################")
    print("Total Accuracy, compared to Chance Level at 42")
    print("####################################################################")
    one_sample_chance(data["C1_Total_Acc"].astype(float)/84*100, 50)
    one_sample_chance(data["C2_Total_Acc"].astype(float)/84*100, 50)
    one_sample_chance(data["C3_Total_Acc"].astype(float)/84*100, 50)
    one_sample_chance(data["C4_Total_Acc"].astype(float)/84*100, 50)


def anova():
    for_anova = pd.read_csv("C:/Users/Herrick Fung/Desktop/Course Materials/Sem 4.1/\
PSY402 Research Thesis II/part1_result/statsmodels.csv", delimiter = ",")
    print("\n####################################################################")
    print("Accurcy: Two-Way ANOVA Repeated Measures")
    print("####################################################################")
    acc_2way = pingouin.rm_anova(for_anova, dv = 'Accuracy', within = ['Pre_Cue', 'Post_Cue'], subject = 'Parti_No', effsize='np2')
    print(acc_2way)
    print("\n####################################################################")
    print("Latency: Two-Way ANOVA Repeated Measures")
    print("####################################################################")
    lat_2way = pingouin.rm_anova(for_anova, dv = 'Latency', within = ['Pre_Cue', 'Post_Cue'], subject = 'Parti_No', effsize='np2')
    print(lat_2way)
    print("\n####################################################################")
    print("R-Square: Two-Way ANOVA Repeated Measures")
    print("####################################################################")
    r_2way = pingouin.rm_anova(for_anova, dv = 'R_Square', within = ['Pre_Cue', 'Post_Cue'], subject = 'Parti_No', effsize='np2')
    print(r_2way)


def graph(data):
    #  Plot 1 - Acc, Lat, R2 over Conditions
    #  Setup for Data
    Mean_Total_Acc = [data["C1_Total_Acc"].astype(float).mean()/84*100,
                      data["C2_Total_Acc"].astype(float).mean()/84*100,
                      data["C3_Total_Acc"].astype(float).mean()/84*100,
                      data["C4_Total_Acc"].astype(float).mean()/84*100,
                     ]
    Std_Total_Acc = [data["C1_Total_Acc"].astype(float).sem()/84*100,
                     data["C2_Total_Acc"].astype(float).sem()/84*100,
                     data["C3_Total_Acc"].astype(float).sem()/84*100,
                     data["C4_Total_Acc"].astype(float).sem()/84*100,
                    ]

    Mean_Latency = [data["C1_Adj_Mean_Lat"].astype(float).mean()*1000,
                    data["C2_Adj_Mean_Lat"].astype(float).mean()*1000,
                    data["C3_Adj_Mean_Lat"].astype(float).mean()*1000,
                    data["C4_Adj_Mean_Lat"].astype(float).mean()*1000,
                   ]
    Std_Latency = [data["C1_Adj_Mean_Lat"].astype(float).sem()*1000,
                   data["C2_Adj_Mean_Lat"].astype(float).sem()*1000,
                   data["C3_Adj_Mean_Lat"].astype(float).sem()*1000,
                   data["C4_Adj_Mean_Lat"].astype(float).sem()*1000,
                  ]

    Mean_R = [data["C1_R_Squared"].astype(float).mean(),
              data["C2_R_Squared"].astype(float).mean(),
              data["C3_R_Squared"].astype(float).mean(),
              data["C4_R_Squared"].astype(float).mean(),
             ]
    Std_R = [data["C1_R_Squared"].astype(float).sem(),
             data["C2_R_Squared"].astype(float).sem(),
             data["C3_R_Squared"].astype(float).sem(),
             data["C4_R_Squared"].astype(float).sem(),
            ]

    #  Set up for Plot 1
    labels = [r'Single $\to$ Single', r'Single $\to$ Set', r'Set $\to$ Set', r'Set $\to$ Single']
    font = font_manager.FontProperties(family='CMU Serif',
                                       size='10')
    fig, host = plt.subplots(figsize=(5,4))
    # fig.subplots_adjust(right=0.75)
    x_pos = np.arange(len(labels))
    width = 0.075
    # par1 = host.twinx()
    # par2 = host.twinx()
    # par2.spines["right"].set_position(("axes", 1.12))

    #  set label
    host.set_xlabel("Experimental Conditions", fontname="CMU Serif", fontsize=12, fontweight="bold")
    host.set_ylabel("Correct Percentage (%)", fontname="CMU Serif", fontsize=12, fontweight="bold")
    # host.set_ylabel("Response Latency (msec)", fontname="CMU Serif", fontsize=12, fontweight="bold")
    # host.set_ylabel(r"${R^2}$", fontname="CMU Serif", fontsize=12, fontweight="bold")

    #  plot
    host.bar([0.0,0.2,0.4,0.6], Mean_Total_Acc, yerr=Std_Total_Acc, width=width, capsize=2, color='#888888')
    # host.bar([0.0,0.2,0.4,0.6], Mean_Latency, yerr=Std_Latency, width=width, capsize=2, color='#888888')
    # host.bar([0,0.2,0.4,0.6], Mean_R, yerr=Std_R, width=width, capsize=2, color='#888888')

    #  set x axis param
    host.set_xticks([0.0,0.2,0.4,0.6])
    host.set_xticklabels(labels, fontname="CMU Serif", fontsize=10)
    # host.spines['top'].set_linewidth(1.2)
    # host.spines['bottom'].set_linewidth(1.2)

    #  set y accuracy param
    # par2.spines['left'].set_color('red')
    # par2.spines['left'].set_linewidth(1.5)
    host.yaxis.label.set_color('black')
    host.tick_params(axis='y', colors='black')
    host.axes.set_yticks(np.arange(30, 105, 10))
    host.axes.set_ylim([30, 105])

#     #  set y latency param
#     par1.spines['right'].set_color('#4D4DFF')
#     par1.spines['right'].set_linewidth(1.5)
#     par1.yaxis.label.set_color('#4D4DFF')
#     par1.tick_params(axis='y', colors='#4D4DFF')
    # host.axes.set_yticks(np.arange(250, 2250, 250))
    # host.axes.set_ylim([250, 2250])

    # set y r square param
    # par2.spines['right'].set_color('green')
    # par2.spines['right'].set_linewidth(1.5)
    # par2.yaxis.label.set_color('green')
    # par2.tick_params(axis='y', colors='green')
    # host.axes.set_yticks(np.arange(0, 1.1, 0.2))
    # host.axes.set_ylim([0, 1.1])

#     #  Legend
#     red_patch = mpatches.Patch(color='r', label="Correct Percentage (%)")
#     blue_patch = mpatches.Patch(color='#4D4DFF', label="Latency (msec)")
#     green_patch = mpatches.Patch(color='g', label=r"${R^2}$")
#     plt.legend(handles=[red_patch, blue_patch, green_patch], prop=font, loc='upper right', fancybox=True, framealpha=0.4)

    #  draw lines
    host.axhline(50, 0, 1, color='black', linestyle=(0, (1, 5)))

    #  Export Plot 1
    # plot1_name = "C:/Users/Herrick Fung/Desktop/Course Materials/Sem 4.1/\
# PSY402 Research Thesis II/part1_result/graph_acc_lat_r2_over_con.png"
    plot1_name = "graph_acc_lat_r2_over_con.png"
    plt.tight_layout()
    fig.savefig(plot1_name, format='png', dpi=384, transparent=False)
    plt.clf()
    plt.close()


    #  Plot 2 - Single-Set Coherency
    #  setup for plot
    labels = [r'Single $\to$ Single', r'Single $\to$ Set', r'Set $\to$ Single']
    font = font_manager.FontProperties(family='CMU Serif',
                                       size='12')
    fig, host = plt.subplots(figsize=(7,5))
    x_pos = np.arange(len(labels))
    width = 0.15

    #  setup data
    Mean_Cohe_Acc = [data["C1_Cohe_Total_Acc"].astype(float).mean()/36*100,
                     data["C2_Cohe_Total_Acc"].astype(float).mean()/36*100,
                     data["C4_Cohe_Total_Acc"].astype(float).mean()/36*100,
                    ]
    Std_Cohe_Acc = [data["C1_Cohe_Total_Acc"].astype(float).sem()/36*100,
                    data["C2_Cohe_Total_Acc"].astype(float).sem()/36*100,
                    data["C4_Cohe_Total_Acc"].astype(float).sem()/36*100,
                   ]

    Mean_InCohe_Acc = [data["C1_InCohe_Total_Acc"].astype(float).mean()/36*100,
                       data["C2_InCohe_Total_Acc"].astype(float).mean()/36*100,
                       data["C4_InCohe_Total_Acc"].astype(float).mean()/36*100,
                      ]
    Std_InCohe_Acc = [data["C1_InCohe_Total_Acc"].astype(float).sem()/36*100,
                      data["C2_InCohe_Total_Acc"].astype(float).sem()/36*100,
                      data["C4_InCohe_Total_Acc"].astype(float).sem()/36*100,
                     ]

    #  set label
    host.set_xlabel("Experimental Conditions", fontname="CMU Serif", fontsize=14, fontweight="bold")
    host.set_ylabel("Correct Percentage (%)", fontname="CMU Serif", fontsize=14, fontweight="bold")

    #  plot
    host.bar(x_pos - width/2, Mean_Cohe_Acc, yerr=Std_Cohe_Acc, width=width, capsize=2, color='#666666')
    host.bar(x_pos + width/2, Mean_InCohe_Acc, yerr=Std_InCohe_Acc, width=width, capsize=2, color='#BBBBBB')

    #  set x axis param
    host.set_xticks(x_pos)
    host.set_xticklabels(labels, fontname="CMU Serif", fontsize=14)
    host.spines['top'].set_linewidth(1.2)
    host.spines['bottom'].set_linewidth(1.2)
    host.spines['right'].set_linewidth(1.2)

    #  set y accuracy param
    host.spines['left'].set_linewidth(1.5)
    host.axes.set_yticks(np.arange(20, 105, 10))
    host.axes.set_ylim([20, 105])

    #  Legend
    mag_patch = mpatches.Patch(color='#666666', label="Coherent")
    orange_patch = mpatches.Patch(color='#BBBBBB', label="Incoherent")
    plt.legend(handles=[mag_patch, orange_patch], prop=font, loc='upper right', fancybox=True, framealpha=0.8)

    #  draw lines
    host.axhline(50, 0, 1, color='black', linestyle=(0, (1, 5)))

    # Export Plot 2
    # plot2_name = "C:/Users/Herrick Fung/Desktop/Course Materials/Sem 4.1/\
# PSY402 Research Thesis II/part1_result/graph_cohe_vs_incohe.png"
    plot2_name = "graph_cohe_vs_incohe.png"
    plt.tight_layout()
    plt.savefig(plot2_name, format='png', dpi=384, transparent=False)
    plt.clf()
    plt.close()


def main():
    data = read_merge_output()
    sys.stdout = open(output_stats_file_name, "w")
    # anova()
    analysis(data)
    graph(data)
    sys.stdout.close()


if __name__ == "__main__":
    main()
