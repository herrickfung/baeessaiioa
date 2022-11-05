'''
Input file: /result/sorted/[Parti_Name]/*.txt
Interim file: /result/fitted_data/univariate.csv
Output file: /result/
#
The script will read the raw csv generated from eeglab after preprocessing,
and sorted by condition, work on univariate analysis, and generate amplitude
and latency.
'''

from scipy.signal import argrelextrema
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import pathlib
import pingouin as pg
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

def manage_path():
    # in
    current_path = pathlib.Path(__file__).parent.absolute()
    raw_path = current_path / 'Cz_as_reference/sorted_by_cond_and_correct_trials_new_ref'
    parti_path_array = [x for x in raw_path.iterdir()]
    return parti_path_array


def read_files(path, condition):
    # 0 - highSD, 1 - lowSD, 2 - Single
    visual_electrode = ["E124", "E125", "E126", "E136", "E137", "E138", "E147", "E148", "E149"]
    frontal_electrode = ["E6", "E7", "E8", "E14", "E15", "E16", "E21", "E22", "E23"]
    parti_array = []
    for j in range(176):
        parti_array.append(str(path.name))

    # pool electrode
    for i in range(len(condition)):
        condition[i] = pd.read_csv(condition[i], delimiter=',', index_col = 0)
        visual_average = np.average([condition[i][e] for e  in visual_electrode], axis = 0)
        frontal_average = np.average([condition[i][e] for e in frontal_electrode], axis = 0)
        condition[i]["visual"] = visual_average
        condition[i]["frontal"] = frontal_average
        condition[i]["parti_info"] =  parti_array

    return condition


def univariate(condition):
    # 0 - highSD, 1 - lowSD, 2 - Single
    visual = [[],[],[]]
    frontal = [[],[],[]]
    np.array(visual)
    np.array(frontal)

    parti_info = []
    visual_peak = []
    visual_latency = []
    frontal_peak = []
    frontal_latency = []

    # time locked and select the averaged channel
    for i in range(len(condition)):
        visual[i] = condition[i].loc[160:280]["visual"].to_numpy()
        frontal[i] = condition[i].loc[160:300]["frontal"].to_numpy()

        #  find local maxima and latency (visual --> frontal)
        visual_local_array = argrelextrema(visual[i], np.less)
        visual_min = np.min(visual[i][visual_local_array])
        visual_peak.append(visual_min)
        visual_late = condition[i].loc[160:280]["visual"][condition[i].loc[160:280]["visual"] == visual_peak[i]].index.tolist()
        visual_latency.append(visual_late[0])

        frontal_local_array = argrelextrema(frontal[i], np.greater)
        frontal_max = np.max(frontal[i][frontal_local_array])
        frontal_peak.append(frontal_max)
        frontal_late = condition[i].loc[160:300]["frontal"][condition[i].loc[160:300]["frontal"] == frontal_peak[i]].index.tolist()
        frontal_latency.append(frontal_late[0])

        parti_info.append(condition[i]["parti_info"][0])

    outputfile = pd.DataFrame({'parti_info': parti_info,
                               'condition': ["High_SD", "Low_SD", "Single"],
                               'N2_Amp': visual_peak,
                               'N2_Lat': visual_latency,
                               'FSP_Amp': frontal_peak,
                               'FSP_Lat': frontal_latency,
                               })
    return outputfile


def stat_analysis(data):

    # Analysis procedure
    # Sphercity test
    n2_amp_sph = pg.sphericity(data, dv = 'N2_Amp', within = 'condition',
                               subject = 'parti_info', method = 'mauchly',
                               alpha = 0.05)
    n2_lat_sph = pg.sphericity(data, dv = 'N2_Lat', within = 'condition',
                               subject = 'parti_info', method = 'mauchly',
                               alpha = 0.05)
    fsp_amp_sph = pg.sphericity(data, dv = 'FSP_Amp', within = 'condition',
                               subject = 'parti_info', method = 'mauchly',
                               alpha = 0.05)
    fsp_lat_sph = pg.sphericity(data, dv = 'FSP_Lat', within = 'condition',
                               subject = 'parti_info', method = 'mauchly',
                               alpha = 0.05)

    # One way anova and posthoc
    n2_amp_anova = pg.rm_anova(data, dv = 'N2_Amp', within = "condition",
                               subject = "parti_info", detailed = True,
                               effsize = 'np2')
    n2_lat_anova = pg.rm_anova(data, dv = 'N2_Lat', within = "condition",
                               subject = "parti_info", detailed = True,
                               effsize = 'np2')
    fsp_amp_anova = pg.rm_anova(data, dv = 'FSP_Amp', within = "condition",
                               subject = "parti_info", detailed = True,
                               effsize = 'np2')
    fsp_lat_anova = pg.rm_anova(data, dv = 'FSP_Lat', within = "condition",
                               subject = "parti_info", detailed = True,
                               effsize = 'np2')

    # Post-Hoc with Bonferroni
    n2_amp_posthoc = pg.pairwise_ttests(data, dv = 'N2_Amp', within = 'condition',
                                      subject = 'parti_info', padjust = 'holm',
                                      effsize = 'cohen')
    n2_lat_posthoc = pg.pairwise_ttests(data, dv = 'N2_Lat', within = 'condition',
                                      subject = 'parti_info', padjust = 'holm',
                                      effsize = 'cohen')
    fsp_amp_posthoc = pg.pairwise_ttests(data, dv = 'FSP_Amp', within = 'condition',
                                      subject = 'parti_info', padjust = 'holm',
                                      effsize = 'cohen')
    fsp_lat_posthoc = pg.pairwise_ttests(data, dv = 'FSP_Lat', within = 'condition',
                                      subject = 'parti_info', padjust = 'holm',
                                      effsize = 'cohen')


    def des(var):
        return [var.name, var.astype(float).mean(), var.astype(float).std()]

    # create the output file
    print("*******************************************************************")
    print("N2 Amplitude Results")
    print("*******************************************************************")
    print(n2_amp_sph)
    print("ANOVA")
    print(n2_amp_anova)
    print("Post Hoc Bonferroni")
    print(n2_amp_posthoc)
    print("*******************************************************************")

    print()
    print("*******************************************************************")
    print("N2 Latency Results")
    print("*******************************************************************")
    print(n2_lat_sph)
    print("ANOVA")
    print(n2_lat_anova)
    print("Post Hoc Bonferroni")
    print(n2_lat_posthoc)
    print("*******************************************************************")

    print()
    print("*******************************************************************")
    print("FSP Amplitude Results")
    print("*******************************************************************")
    print(fsp_amp_sph)
    print("ANOVA")
    print(fsp_amp_anova)
    print("Post Hoc Bonferroni")
    print(fsp_amp_posthoc)
    print("*******************************************************************")

    print()
    print("*******************************************************************")
    print("FSP Latency Results")
    print("*******************************************************************")
    print(fsp_lat_sph)
    print("ANOVA")
    print(fsp_lat_anova)
    print("Post Hoc Bonferroni")
    print(fsp_lat_posthoc)
    print("*******************************************************************")


def stat_graph(data):
    highsd = data[data["condition"] == "High_SD"]
    lowsd = data[data["condition"] == "Low_SD"]
    single = data[data["condition"] == "Single"]

    # N2 Plot
    plt.clf()
    mean_n2_amp = [single["N2_Amp"].astype(float).mean(),
                   lowsd["N2_Amp"].astype(float).mean(),
                   highsd["N2_Amp"].astype(float).mean()]
    std_n2_amp = [single["N2_Amp"].astype(float).sem(),
                  lowsd["N2_Amp"].astype(float).sem(),
                  highsd["N2_Amp"].astype(float).sem()]
    mean_n2_lat = [single["N2_Lat"].astype(float).mean(),
                   lowsd["N2_Lat"].astype(float).mean(),
                   highsd["N2_Lat"].astype(float).mean()]
    std_n2_lat = [single["N2_Lat"].astype(float).sem(),
                  lowsd["N2_Lat"].astype(float).sem(),
                  highsd["N2_Lat"].astype(float).sem()]

    labels = ['Single Element', 'LowSD-Set', 'HighSD-Set']
    font = font_manager.FontProperties(family='CMU Serif',
                                       size='10')
    fig, host = plt.subplots(figsize=(4.5,6))
    x_pos = np.array([0.0, 0.1, 0.2])
    width = 0.05

    host.set_xlabel("Experimental Condition", fontname="CMU Serif", fontsize=12, fontweight='bold')
    host.set_ylabel(r'N170 Peak Amplitude ($\mu$V)', fontname="CMU Serif", fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    host.bar(x_pos, mean_n2_amp, yerr=(std_n2_amp,[0,0,0]), width = width, capsize = 2, color = "#666666", ecolor='k')
    host.set_xticks(x_pos)
    host.set_xticklabels(labels, fontname='CMU Serif', fontsize=10)
    host.axes.set_ylim([0, -16])
    host.axes.set_yticks(np.arange(-16, 0, 2))

    par1 = host.twinx()
    par1.set_ylabel('Peak Latency (ms)', fontname='CMU Serif', fontsize=12, fontweight='bold')
    par1.plot(x_pos+0.02, mean_n2_lat, color='#666666', marker='^', mfc='#666666', mec='#666666', markersize=8)
    par1.errorbar(x_pos+0.02, mean_n2_lat, yerr=(std_n2_lat, [0,0,0]),
                  color = "#666666", elinewidth=1, ecolor='k', capsize=2)
    par1.axes.set_ylim([0, 250])

    plot1_name = "N2_over_con.png"
    plt.tight_layout()
    fig.savefig(plot1_name, format='png', dpi=384, transparent=False)
    plt.clf()
    plt.close()

    # FSP Plot
    mean_fs_amp = [single["FSP_Amp"].astype(float).mean(),
                   lowsd["FSP_Amp"].astype(float).mean(),
                   highsd["FSP_Amp"].astype(float).mean()]
    std_fs_amp = [single["FSP_Amp"].astype(float).sem(),
                  lowsd["FSP_Amp"].astype(float).sem(),
                  highsd["FSP_Amp"].astype(float).sem()]
    mean_fs_lat = [single["FSP_Lat"].astype(float).mean(),
                   lowsd["FSP_Lat"].astype(float).mean(),
                   highsd["FSP_Lat"].astype(float).mean()]
    std_fs_lat = [single["FSP_Lat"].astype(float).sem(),
                  lowsd["FSP_Lat"].astype(float).sem(),
                  highsd["FSP_Lat"].astype(float).sem()]

    fig, host = plt.subplots(figsize=(4.5,6))

    host.set_xlabel("Experimental Condition", fontname="CMU Serif", fontsize=12, fontweight='bold')
    host.set_ylabel(r'FSP Peak Amplitude ($\mu$V)', fontname="CMU Serif", fontsize=12, fontweight='bold')
    host.bar(x_pos, mean_fs_amp, yerr=([0,0,0], std_fs_amp), width = width, capsize = 2, color = "#666666", ecolor='k')
    host.set_xticks(x_pos)
    host.set_xticklabels(labels, fontname='CMU Serif', fontsize=10)
    host.axes.set_ylim([0, 8])
    host.axes.set_yticks(np.arange(0, 8, 1))

    par1 = host.twinx()
    par1.set_ylabel('Peak Latency (ms)', fontname='CMU Serif', fontsize=12, fontweight='bold')
    par1.plot(x_pos+0.02, mean_n2_lat, color = '#666666', marker='^', mfc='#666666', mec='#666666', markersize=8)
    par1.errorbar(x_pos+0.02, mean_n2_lat, yerr=(std_n2_lat, [0,0,0]),
                  color='#666666', ecolor='k', elinewidth=1, capsize=2)
    par1.axes.set_ylim([0, 250])

    plot2_name = "FSP_over_con.png"
    plt.tight_layout()
    fig.savefig(plot2_name, format='png', dpi=384, transparent=False)
    plt.clf()
    plt.close()


def main():
     in_path = manage_path()
     i = 0
     for path in in_path:
         eeg_file = [e for e in path.iterdir() if e.match('*.txt')]
         eeg_data = read_files(path, eeg_file)
         ind_result = univariate(eeg_data)
         if i < 1:
             merged_result = ind_result
         else:
             merged_result = pd.concat([merged_result, ind_result])
         i = i + 1

    # Export the raw data before stats.
    univariate_filename = "~/Desktop/Course Materials/Sem 4.1/\PSY402 Research Thesis II/experiment_part2/result/fitted_data/univariate.csv"
    merged_result.to_csv(univariate_filename, sep=',', index=False)

    merged_result = pd.read_csv("~/Desktop/Course Materials/Sem 4.1/PSY402 Research Thesis II/experiment_part2/result/fitted_data/univariate.csv", ',')

    stat_output_filename = "univariate_stat.txt"
    sys.stdout = open(stat_output_filename, 'w')
    stat_analysis(merged_result)
    sys.stdout.close()
    stat_graph(merged_result)


if __name__ == "__main__":
    main()
