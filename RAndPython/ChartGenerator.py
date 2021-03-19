import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.patches import Rectangle


import plotly.express as px
import tikzplotlib

from scipy.stats import linregress
def fit_line1(x, y):
    """Return slope, intercept of best fit line."""
    # Remove entries where either x or y is NaN.
    clean_data = pd.concat([x, y], 1).dropna(0) # row-wise
    (_, x), (_, y) = clean_data.iteritems()
    slope, intercept, r, p, stderr = linregress(x, y)
    return slope, intercept # could also return stderr

import statsmodels.api as sm
def fit_line2(x, y):
    """Return slope, intercept of best fit line."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X, missing='drop') # ignores entires where x or y is NaN
    fit = model.fit()
    return fit.params[1], fit.params[0] # could also return stderr in each via fit.bse


def log_log_slope(df,x,y):
    target_data = df[[x,y]]
    target_data = np.log10(target_data)
    (slope,intercept)=fit_line1(target_data[x],target_data[y])
    #print(slope,intercept)
    return slope


def main():
    # matplotlib.use("pgf")
    sns.set_style('whitegrid')
    sns.set_context("paper")
    matplotlib.rcParams.update({
        #        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        #       'pgf.rcfonts': False,
    })


    displayNameDict = {
    'CPULinearQuad':    'Linear/Quad/CPU',
    'CPUBicubicQuad':   'Bicubic/Quad/CPU',
    'CPUBicubicSparse': 'Bicubic/Sparse MM/CPU',
    'GPUBicubicSparse': 'Bicubic/Sparse MM/GPU',
    'CPUDCTPadded' :    'Padded FFT/CPU',
    'CPUChebDense':     'Cheb/Dense MM/CPU',
    'GPUChebDense':     'Cheb/Dense MM/GPU',
    'CPULinearSparse':  'Linear/Sparse MM/CPU',
    'GPULinearSparse':  'Linear/Sparse MM/GPU',
    'NonsmoothAbs' :    'non-smooth ridge',
    'NonsmoothSqrt' :   'non-smooth horn',
    'SmoothExp' :       'smooth exponential',
    'SmoothRunge' :     "smooth bivariate runge"
    }

#read all of our data
    data_all = pd.read_csv('temp.csv')

#We decided to filter out some data that didn't add to the paper.
#in particular, float vs double was surprisingly not a big deal.
    data_all = data_all[data_all['calculator'] != 'CPUDCTnopad']
    data_all = data_all[data_all['functionName'] != 'SmoothPoly']
    data_all = data_all[data_all['functionName'] != 'NonsmoothRungeAbs']
    data_all = data_all[data_all['N'] != 8]
    data_double = data_all[data_all['bytes']==8]
    data_float = data_all[data_all['bytes']==4]

#below code prepares speed impact of single vs float and GPU vs CPU where applicable.
    HzOnly = data_all[data_all['calculator'] != 'CPUDCTnopad']
    HzOnly = HzOnly[HzOnly['calculator'] != 'CPUBicubicQuad']
    HzOnly = HzOnly[HzOnly['calculator'] != 'CPULinearQuad']
    HzOnly = HzOnly.groupby(['N','calculator','bytes']).agg(speed=('calcHz', 'median')).reset_index()
    HzOnlyWide = HzOnly.pivot(index=('calculator','N'),columns = 'bytes', values='speed');
    HzOnlyWide['ratio'] = HzOnlyWide[4] / HzOnlyWide[8]  #drop NA later
    HzOnlyWide2 = HzOnly[HzOnly['bytes'] == 8]
    HzOnlyWide2 = HzOnlyWide2[HzOnlyWide2['N'] < 290]
    RatioChart = HzOnlyWide2.pivot(index=('N'),columns='calculator', values='speed')
    RatioChart['Linear'] = RatioChart['GPULinearSparse'] /  RatioChart['CPULinearSparse']
    RatioChart['Bicubic'] = RatioChart['GPUBicubicSparse'] /  RatioChart['CPUBicubicSparse']
    RatioChart['Chebyshev'] = RatioChart['GPUChebDense'] /  RatioChart['CPUChebDense']
    RatioChart = RatioChart[['Linear','Bicubic','Chebyshev']]
    RatioChart = RatioChart.melt(ignore_index=False).dropna()

    SpeedVsN = data_double
    SpeedVsN= SpeedVsN.groupby(['calculator','N']) .agg(speed=('calcHz','median')).reset_index()

    regSpeedVsErr = data_double.replace(displayNameDict)
    regSpeedVsErr= regSpeedVsErr.groupby(['functionName', 'calculator'])

#This section will prepare the speed study as a function of N.
    interpDict = {
        'CPULinearQuad': 'Linear',
        'CPUBicubicQuad': 'Bicubic',
        'CPUBicubicSparse': 'Bicubic',
        'GPUBicubicSparse': 'Bicubic',
        'CPUDCTPadded': 'Padded FFT',
        'CPUChebDense': 'Chebyshev',
        'GPUChebDense': 'Chebyshev',
        'CPULinearSparse': 'Linear',
        'GPULinearSparse': 'Linear',
    }

    algoDict = {
        'CPULinearQuad': 'Quadrature',
        'CPUBicubicQuad': 'Quadrature',
        'CPUBicubicSparse': 'Sparse CPU',
        'GPUBicubicSparse': 'Sparse GPU',
        'CPUDCTPadded': 'FFT',
        'CPUChebDense': 'Dense CPU',
        'GPUChebDense': 'Dense GPU',
        'CPULinearSparse': 'Sparse CPU',
        'GPULinearSparse': 'Sparse GPU',
    }

    RatioChart = RatioChart.replace(interpDict)
    RatioChart = RatioChart.replace(displayNameDict)
    #print(RatioChart)
    RatioChart = RatioChart[(RatioChart['calculator'] != 'Linear') | (RatioChart['value'] < 7) ] #one bad data point

    GP = sns.relplot(data=RatioChart, x='N', y='value', hue='calculator',col='calculator', facet_kws={"sharey": False, "sharex": False},height=3,
                     col_wrap=2,legend=False)
    plt.autoscale(True)

    GP.set_axis_labels("$N$", "Speedup from GPU acceleration")
    GP.set_titles('{col_name}')
    plt.savefig(f"GPUAccel.pdf", bbox_inches='tight')
    #plt.show()

    regSpeedVsN = SpeedVsN.groupby(['calculator']).apply(lambda x: log_log_slope (x, 'N', 'speed'))
    myfig,myax = plt.subplots(ncols=1)
    g = sns.relplot(data=SpeedVsN, x='N', y='speed', hue='calculator', kind='line', style='calculator',   legend='full', )

    plt.autoscale(True)
    g.ax.set_xscale("log", base=2)
    g.ax.set_yscale("log")
    g.ax.xaxis.grid(True, "minor", linewidth=.1)
    g.ax.yaxis.grid(True, "minor", linewidth=.1)
    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    # formatter.set_powerlimits((-1, 1))
    g.ax.xaxis.set_major_formatter(formatter)
    g.ax.set_xlabel("$N$ ($\log_2$ scale)")
    g.ax.set_ylabel("Calcs per second ($\log_{10}$ scale)")
    g.ax.set_title("Speed of gyroaveraging vs input grid size")

    g.legend.set_title("Algorithm")
    g.legend.set_visible(False)

    handles, labels = g.ax.get_legend_handles_labels()
    algocount = len(handles)
    extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]
    label_empty=[""]
    interpNames = list(map(interpDict.get,labels))
    algoNames = list(map(algoDict.get,labels))
    slopeDict  = regSpeedVsN.to_dict()
    slopes = list(map(lambda n: '%.2f'%n,(map(slopeDict.get,labels))))

    new_legend_handles = extra + handles + (extra *(algocount*3+3))
    new_legend_labels= label_empty*(algocount+1)+["Interp"]+interpNames+["Algorithm"]+algoNames+["Slope"]+slopes
    plt.legend(new_legend_handles,new_legend_labels,bbox_to_anchor=(0.,-0.15,1.,0.0),loc='upper center',  borderaxespad=0.2,ncol=4, handletextpad=-0.2,
               fontsize="x-small")
    plt.subplots_adjust(bottom=0.33)
    plt.savefig(f"SpeedVsN.eps", bbox_inches='tight' )

#Next we save CSVs containing regression coefficients; we may not need these for the paper.
    regSpeedVsN.to_csv('RegSpeedVsN.csv')
    regSpeedVsErr = regSpeedVsErr.apply(lambda x: log_log_slope (x, 'N', 'maxError',)).unstack()
    regSpeedVsErr.to_csv("regSpeedVsErr.csv")


    speedData = data_double[data_double['calculator'] != 'CPULinearQuad']
    speedData = speedData[speedData['calculator'] != 'CPUBicubicQuad']

    #sns.relplot(data=HzOnlyWide,x='N',y='ratio',hue='calculator')
    #plt.show()




#The below section is going to study error convergence vs N for each interp method.
    tempdata = data_double
    tempdata = tempdata[tempdata['calculator'] != 'CPULinearQuad']
    tempdata = tempdata[tempdata['calculator'] != 'CPUBicubicQuad']

    tempdata = tempdata.replace(interpDict)
    tempdata = tempdata.replace(displayNameDict)

    g = sns.relplot(data=tempdata, x='N', y='maxError', hue='calculator', kind='line', style='calculator',
                    facet_kws={"sharey": False, "sharex": False, "legend_out": False}, col='functionName', col_wrap=2,
                    legend='auto', height=3)
    plt.autoscale(True)

    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    # formatter.set_powerlimits((-1, 1))
    for ax in g.axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.grid(True, "minor", linewidth=.1)
        ax.yaxis.grid(True, "minor", linewidth=.1)
        ax.xaxis.set_major_formatter(formatter)

    g.set_axis_labels("$N$", "$L_\infty$ error")
    g.set_titles('{col_name}')
    g.legend.set_title("Interp")
    g.legend.set_bbox_to_anchor((1.2, -1.2))

    handles, labels = g.axes[0].get_legend_handles_labels()

    g.add_legend(dict(zip(labels, handles)), title="Interp", bbox_to_anchor=(1.05, -1.4), loc='upper center',
                 fontsize="x-small", ncol=4)
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f"ConvergenceAllNew.pdf", bbox_inches='tight' )

    regErrVsN = tempdata.groupby(['functionName', 'calculator'])
    regErrVsN = regErrVsN.apply(lambda x: log_log_slope(x, 'N', 'maxError', )).unstack()
    #regErrVsN = regErrVsN.agg(err=('maxError', 'median')).reset_index()
    regErrVsN.to_csv("regErrVsN.csv")

#the below section is going to plot speed vs accuracy
    tempdata2 = data_double
    tempdata2 = tempdata2[tempdata2['calculator'] != 'CPULinearQuad']
    tempdata2 = tempdata2[tempdata2['calculator'] != 'CPUBicubicQuad']

    tempdata2 = tempdata2.replace(displayNameDict)

    g = sns.relplot(data=tempdata2, x='calcHz', y='maxError', hue='calculator', kind='line', style='calculator',
                    facet_kws={"sharey": False, "sharex": False, "legend_out": False}, col='functionName', col_wrap=2,
                    legend='auto', height=3)
    plt.autoscale(True)

    from matplotlib import ticker
    #formatter = ticker.ScalarFormatter(useMathText=True)
    #formatter.set_scientific(False)
    # formatter.set_powerlimits((-1, 1))
    for ax in g.axes:
        ax.set_xscale("log" )
        ax.set_yscale("log")
        ax.xaxis.grid(True, "minor", linewidth=.1)
        ax.yaxis.grid(True, "minor", linewidth=.1)
    #    ax.xaxis.set_major_formatter(formatter)

    g.set_axis_labels("Calcs per second", "$L_\infty$ error")
    g.set_titles('{col_name}')
    g.legend.set_title("Algorithm")

    handles, labels = g.axes[0].get_legend_handles_labels()

    g.add_legend(dict(zip(labels, handles)), title="Algorithm", bbox_to_anchor=(1.05, -1.4), loc='upper center',
                 fontsize="x-small", ncol=3)
    plt.subplots_adjust(bottom=0.2)
    #plt.show()
    plt.savefig(f"OptimalAllNew.pdf", bbox_inches='tight')

 #TEMP
    exit()
    for m in models:
        tempdata = data_double.loc[data_double['functionName'] == m]

        tempdata = tempdata[tempdata['calculator'] != 'CPULinearQuad']
        tempdata = tempdata[tempdata['calculator'] != 'CPUBicubicQuad']

        tempdata = tempdata.replace(displayNameDict)
        plt.rc('text', usetex=True)
        plt.rc('font', family="serif")
        plt.rc('figure', figsize=(5, 5))
        g = sns.relplot(data=tempdata, x='calcHz', y='maxError', hue='calculator', kind='line', style='calculator',
                    facet_kws={"sharey": False, "sharex": False})
        plt.autoscale(True)
        # for ax in g.axes:
        g.ax.set_xscale("log")
        g.ax.set_yscale("log")
        g.ax.xaxis.grid(True, "minor", linewidth=.2)
        g.ax.yaxis.grid(True, "minor", linewidth=.2)

        g.ax.set_xlabel("Calcs per second ($\log$ scale)")
        g.ax.set_ylabel("Maximum Error ($\log$ scale)")
        g.ax.set_title(f"Gyroaveraging error vs wall clock time for {displayNameDict[m]}")
        g.legend.set_title("Algorithm")

        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)
        # formatter.set_powerlimits((-1, 1))
        g.ax.xaxis.set_major_formatter(formatter)
        plt.savefig(f"Optimal{m}.eps", bbox_inches='tight')

    exit()
#the below code does individual convergence plots, we don't need it.
# funcs = uniqFuncs = data_double.functionName.dropna().unique();
# for m in funcs:
#     tempdata = data_double.loc[data_double['functionName'] == m]
#     tempdata = tempdata[tempdata['calculator'] != 'CPULinearQuad']
#     tempdata = tempdata[tempdata['calculator'] != 'CPUBicubicQuad']
#
#     tempdata = tempdata.replace(interpDict)
#     plt.rc('text', usetex=True)
#     plt.rc('font', family="serif")
#     # plt.rc('figure', figsize=(5,5))
#     g = sns.relplot(data=tempdata, x='N', y='maxError', hue='calculator', kind='line', style='calculator',
#                     facet_kws={"sharey": False, "sharex": False})
#     plt.autoscale(True)
#     # for ax in g.axes:
#     g.ax.set_xscale("log", base=2)
#     g.ax.set_yscale("log")
#     g.ax.xaxis.grid(True, "minor", linewidth=.1)
#     g.ax.yaxis.grid(True, "minor", linewidth=.1)
#
#     g.ax.set_xlabel("$N$ ($\log_2$ scale)")
#     g.ax.set_ylabel("$L_\infty$ error ($\log_{10}$ scale)")
#     g.ax.set_title(f"Gyroaveraging error convergence vs  grid size for {displayNameDict[m]}")
#     g.legend.set_title("Algorithm")
#
#     from matplotlib import ticker
#
#     formatter = ticker.ScalarFormatter(useMathText=True)
#     formatter.set_scientific(False)
#     # formatter.set_powerlimits((-1, 1))
#     g.ax.xaxis.set_major_formatter(formatter)
#     # plt.show()
#     plt.savefig(f"Convergence{m}.eps", bbox_inches='tight')
#     funcs = uniqFuncs = data_double.functionName.dropna().unique();
#     for m in funcs:
#         tempdata = data_double.loc[data_double['functionName'] == m]
#         tempdata = tempdata[tempdata['calculator'] != 'CPULinearQuad']
#         tempdata = tempdata[tempdata['calculator'] != 'CPUBicubicQuad']
#
#         tempdata = tempdata.replace(interpDict)
#         plt.rc('text', usetex=True)
#         plt.rc('font', family="serif")
#         # plt.rc('figure', figsize=(5,5))
#         g = sns.relplot(data=tempdata, x='N', y='maxError', hue='calculator', kind='line', style='calculator',
#                         facet_kws={"sharey": False, "sharex": False})
#         plt.autoscale(True)
#         # for ax in g.axes:
#         g.ax.set_xscale("log", base=2)
#         g.ax.set_yscale("log")
#         g.ax.xaxis.grid(True, "minor", linewidth=.1)
#         g.ax.yaxis.grid(True, "minor", linewidth=.1)
#
#         g.ax.set_xlabel("$N$ ($\log_2$ scale)")
#         g.ax.set_ylabel("$L_\infty$ error ($\log_{10}$ scale)")
#         g.ax.set_title(f"Gyroaveraging error convergence vs  grid size for {displayNameDict[m]}")
#         g.legend.set_title("Algorithm")
#
#         from matplotlib import ticker
#
#         formatter = ticker.ScalarFormatter(useMathText=True)
#         formatter.set_scientific(False)
#         # formatter.set_powerlimits((-1, 1))
#         g.ax.xaxis.set_major_formatter(formatter)
#         # plt.show()
#         plt.savefig(f"Convergence{m}.eps", bbox_inches='tight')

    # we don't need the below plot
    # g = sns.relplot(data=data_double, x='N', y='calcHz', hue='calculator', kind='line', style='calculator',
    #                 col='functionName', col_wrap=2, facet_kws={"sharey": False, "sharex": False})
    # plt.autoscale(True)
    # for ax in g.axes:
    #     ax.set_xscale("log", base=2)
    #     ax.set_yscale("log")
    #     ax.xaxis.grid(True, "minor", linewidth=.2)
    #     ax.yaxis.grid(True, "minor", linewidth=.2)
    #     from matplotlib import ticker
    #     formatter = ticker.ScalarFormatter(useMathText=True)
    #     formatter.set_scientific(False)
    #     # formatter.set_powerlimits((-1, 1))
    #     ax.xaxis.set_major_formatter(formatter)
    # plt.show()

    # g = sns.relplot(data=speedData,x='calcHz',y='maxError', hue='calculator', kind='line', style='calculator', col='functionName', col_wrap=2,facet_kws={"sharey":False, "sharex":False})
    # plt.autoscale(True)
    # for ax in g.axes:
    #     ax.set_xscale("log")
    #     ax.set_yscale("log")
    #     ax.xaxis.grid(True, "minor", linewidth=.25)
    #     ax.yaxis.grid(True, "minor", linewidth=.25)
    #     from matplotlib import ticker
    #     formatter = ticker.ScalarFormatter(useMathText=True)
    #     formatter.set_scientific(False)
    #     ax.xaxis.set_major_formatter(formatter)
   # plt.show()

if __name__ == "__main__":
    main()
