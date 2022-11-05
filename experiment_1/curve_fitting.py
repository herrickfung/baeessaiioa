'''
Input file: /naive_result/processed
Output file (Data): /naive_result/fitted_data
Output file (Graph): /naive_result/fitted_graph
#
The script will read the processed data by process.py from the input directory,
and will return a single file containing information about the psy curve,
all gathered in the fitted_data directory. These data can be merged and ready
for statistical analysis.
The fitted_graph will contain graphs for each parti for each condition,
which maybe useful.
'''

from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

input_folder_path =\
    "C:/Users/Herrick Fung/Desktop/Course Materials/Sem 4.1/\
PSY402 Research Thesis II/experiment/naive_result/processed/"
output_data_folder_path =\
    "C:/Users/Herrick Fung/Desktop/Course Materials/Sem 4.1/\
PSY402 Research Thesis II/experiment/naive_result/fitted_data/"
output_graph_folder_path =\
    "C:/Users/Herrick Fung/Desktop/Course Materials/Sem 4.1/\
PSY402 Research Thesis II/experiment/naive_result/fitted_graph/"


def reading(file):
    input_filename = f"processed/{file}"
    input_data = pd.read_csv(input_filename, 'r', delimiter=',')
    Total_Accuracy = []
    Total_Accuracy = input_data.Accuracy_10 +\
        input_data.Accuracy_20 +\
        input_data.Accuracy_30

    Cohe_Total_Acc = []
    Incohe_Total_Acc = []
    Cohe_Total_Acc = input_data.Cohe_Accuracy_10 +\
        input_data.Cohe_Accuracy_20 +\
        input_data.Cohe_Accuracy_30
    Incohe_Total_Acc = input_data.InCohe_Accuracy_10 +\
        input_data.InCohe_Accuracy_20 +\
        input_data.InCohe_Accuracy_30


    unchanged = np.array([input_data.Exp_Date,
                          input_data.Exp_Time,
                          input_data.Parti_Name,
                          input_data.Age,
                          input_data.Gender,
                          input_data.Dominant_Hand,
                          input_data.Condition,
                          input_data.Accuracy_10,
                          input_data.Accuracy_20,
                          input_data.Accuracy_30,
                          Total_Accuracy,
                          input_data.Adj_Latency_Mean,
                          input_data.Cohe_Accuracy_10,
                          input_data.Cohe_Accuracy_20,
                          input_data.Cohe_Accuracy_30,
                          Cohe_Total_Acc,
                          input_data.InCohe_Accuracy_10,
                          input_data.InCohe_Accuracy_20,
                          input_data.InCohe_Accuracy_30,
                          Incohe_Total_Acc
                          ])

    for_fit = pd.read_csv(input_filename, 'r', delimiter=',',
                          usecols=[7, 8, 9, 10, 11, 12, 13],
                          )
    for_fit = for_fit.values
    return unchanged, for_fit


def fitting(unchange, fit_data):
    try:
        os.mkdir(f"{output_graph_folder_path}\
{unchange[0, 0]}{unchange[1, 0]}_{unchange[2, 0]}")
    except FileExistsError:
        pass

    xdata = np.arange(-30, 31, 10)
    alpha_array = []
    beta_array = []
    success_array = []
    r_squared_array = []

    for i in range(4):
        ydata = fit_data[i]
        ydata = ydata / 14

        # Fitting
        def gauss(params):
            beta = params[0]
            alpha = params[1]
            yPred = norm.cdf(xdata, loc=alpha, scale=beta)
            negLL = -np.sum(norm.logpdf(ydata, loc=yPred, scale=1))
            return negLL

        initParams = [1, 1]
        results = minimize(gauss, initParams, method='Nelder-Mead')
        alpha_array.append(results.x[1])
        beta_array.append(results.x[0])
        success_array.append(results.success)
        estimated_param = results.x

        # You can insert your bootstrapping procedure here.

        # Goodness of Fit Procedure (R-Squared)
        Observe_y = ydata
        Expect_y = norm.cdf(xdata, loc=estimated_param[1], scale=estimated_param[0])
        correlation = np.corrcoef(Expect_y, Observe_y)
        r = correlation[0, 1]
        r_squared = r**2
        r_squared_array.append(r_squared)

        # Plotting
        filename = f"{output_graph_folder_path}{unchange[0, 0]}\
{unchange[1, 0]}_{unchange[2, 0]}/Condition_{i + 1}.pdf"
        # filename = output_graph_folder_path +\
        #     str(unchange[0, 0]) +\
        #     str(unchange[1, 0]) +\
        #     "_" + str(unchange[2, 0]) +\
        #     "/Condition_" + str(i + 1) + ".pdf"

        plt.clf()
        xforplot = np.arange(-30, 30.05, 0.05)
        yforplot = norm.cdf(xforplot, loc=estimated_param[1], scale=estimated_param[0])
        plt.plot(xdata, ydata, 'ko')
        plt.plot(xforplot, yforplot, 'k')
        plt.xlim(-35, 35)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("Tilt (Degree in Clockwise)")
        plt.ylabel("Frequency of Responding Clockwise")
        plt.savefig(filename)
        plt.close()

    fit_result_array = [alpha_array,
                        beta_array,
                        success_array,
                        r_squared_array
                        ]
    return fit_result_array


def outputting(unchange, fit):
    output_filename = f"{output_data_folder_path}/fitted_{unchange[0,0]}\
{unchange[1,0]}_{unchange[2,0]}.csv"
    output_data = pd.DataFrame({'Exp_Date': unchange[0],
                                'Exp_Time': unchange[1],
                                'Parti_Name': unchange[2],
                                'Age': unchange[3],
                                'Gender': unchange[4],
                                'Dominant_Hand': unchange[5],
                                'Condition': unchange[6],
                                'Fit_Success': fit[2],
                                'Alpha': fit[0],
                                'Beta': fit[1],
                                'R_Squared': fit[3],
                                'Accuracy_10': unchange[7],
                                'Accuracy_20': unchange[8],
                                'Accuracy_30': unchange[9],
                                'Total_Accuracy': unchange[10],
                                'Adj_Latency_Mean': unchange[11],
                                'Cohe_10_Acc': unchange[12],
                                'Cohe_20_Acc': unchange[13],
                                'Cohe_30_Acc': unchange[14],
                                'Cohe_Total_Acc': unchange[15],
                                'InCohe_10_Acc': unchange[16],
                                'InCohe_20_Acc': unchange[17],
                                'InCohe_30_Acc': unchange[18],
                                'InCohe_Total_Acc': unchange[19]
                                })

    output_data.to_csv(output_filename, sep=',', index=False)


def main():
    # Create output directory
    try:
        os.mkdir(output_data_folder_path)
        os.mkdir(output_graph_folder_path)
        print("Data & Graph Directory Created!")
    except FileExistsError:
        print("Data & Graph Directory Existed!")

    # read files from input directory and process it
    print("Curve Fitted the following files:")
    for filename in os.listdir(input_folder_path):
        if filename.endswith("_ep_experiment.csv"):
            unchange_array, fit_array = reading(filename)
            fit_result = fitting(unchange_array, fit_array)
            outputting(unchange_array, fit_result)
            print(filename)


if __name__ == "__main__":
    main()
