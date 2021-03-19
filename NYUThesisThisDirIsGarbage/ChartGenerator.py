import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px



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
    data_all = pd.read_csv('temp.csv')

    sns.set_style('whitegrid')
    sns.set_context("paper")


    uniqModels = data_all.functionName.dropna().unique();

    data_all = data_all[data_all['calculator'] != 'CPUDCTnopad']
    data_all = data_all[data_all['functionName'] != 'SmoothPoly']
    data_all = data_all[data_all['functionName'] != 'NonsmoothRungeAbs']
    data_all = data_all[data_all['N'] != 8]
    data_double = data_all[data_all['bytes']==8]
    data_float = data_all[data_all['bytes']==4]

    HzOnly = data_all[data_all['calculator'] != 'CPUDCTnopad']
    HzOnly = HzOnly[HzOnly['calculator'] != 'CPUBicubicQuad']
    HzOnly = HzOnly[HzOnly['calculator'] != 'CPULinearQuad']
    HzOnly = HzOnly.groupby(['N','calculator','bytes']).agg(speed=('calcHz', 'median')).reset_index()
    HzOnlyWide = HzOnly.pivot(index=('calculator','N'),columns = 'bytes', values='speed');
    HzOnlyWide['ratio'] = HzOnlyWide[4] / HzOnlyWide[8]  #drop NA later
    HzOnlyWide2 = HzOnly[HzOnly['bytes'] == 8]
    HzOnlyWide2 = HzOnlyWide2[HzOnlyWide2['N'] < 290]
    RatioChart = HzOnlyWide2.pivot(index=('N'),columns='calculator', values='speed')
    RatioChart['LinearSpeedup'] = RatioChart['GPULinearSparse'] /  RatioChart['CPULinearSparse']
    RatioChart['BicubicSpeedup'] = RatioChart['GPUBicubicSparse'] /  RatioChart['CPUBicubicSparse']
    RatioChart['ChebSpeedup'] = RatioChart['GPUChebDense'] /  RatioChart['CPUChebDense']
    RatioChart = RatioChart[['LinearSpeedup','BicubicSpeedup','ChebSpeedup']]
    RatioChart = RatioChart.melt(ignore_index=False).dropna()

    regSpeedVsN = data_double
    regSpeedVsN= regSpeedVsN.groupby(['functionName', 'calculator'])
    regSpeedVsN = regSpeedVsN.apply(lambda x: log_log_slope (x, 'N', 'calcHz',))
    print(regSpeedVsN)

    regSpeedVsErr = data_double
    regSpeedVsErr= regSpeedVsErr.groupby(['functionName', 'calculator'])
    regSpeedVsErr = regSpeedVsErr.apply(lambda x: log_log_slope (x, 'N', 'maxError',))
    print(regSpeedVsErr)



    speedData = data_double[data_double['calculator'] != 'CPULinearQuad']
    speedData = speedData[speedData['calculator'] != 'CPUBicubicQuad']

    #sns.relplot(data=HzOnlyWide,x='N',y='ratio',hue='calculator')
    #plt.show()
    #sns.relplot(data=RatioChart, x='N', y='value', col = 'calculator', facet_kws={"sharey":False, "sharex":False})
    #plt.show()


    models = uniqModels = data_double.functionName.dropna().unique();
    for m in models:
        tempdata = data_double.loc[data_double['functionName']==m]
        g = sns.relplot(data=tempdata,x='N',y='maxError', hue='calculator', kind='line', style='calculator', facet_kws={"sharey":False, "sharex":False})
        plt.autoscale(True)
        #for ax in g.axes:
        g.ax.set_xscale("log", base=2)
        g.ax.set_yscale("log")
        g.ax.xaxis.grid(True, "minor", linewidth=.25)
        g.ax.yaxis.grid(True, "minor", linewidth=.25)
        g.ax.set_xlabel('N')
        g.ax.set_ylabel('Maximum Error')
        g.ax.set_title(m)
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)
        #formatter.set_powerlimits((-1, 1))
        g.ax.xaxis.set_major_formatter(formatter)
        import tikzplotlib
        g.savefig("delme.pdf")
        plt.show()
        tikzplotlib.save(f"delme{m}.tex", encoding='utf-8')



    g = sns.relplot(data=data_double,x='N',y='calcHz', hue='calculator', kind='line', style='calculator', col='functionName', col_wrap=2,facet_kws={"sharey":False, "sharex":False})
    plt.autoscale(True)
    for ax in g.axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.grid(True, "minor", linewidth=.25)
        ax.yaxis.grid(True, "minor", linewidth=.25)
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)
        #formatter.set_powerlimits((-1, 1))
        ax.xaxis.set_major_formatter(formatter)
    #plt.show()


    g = sns.relplot(data=speedData,x='calcHz',y='maxError', hue='calculator', kind='line', style='calculator', col='functionName', col_wrap=2,facet_kws={"sharey":False, "sharex":False})
    plt.autoscale(True)
    for ax in g.axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.grid(True, "minor", linewidth=.25)
        ax.yaxis.grid(True, "minor", linewidth=.25)
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
   # plt.show()

if __name__ == "__main__":
    main()
