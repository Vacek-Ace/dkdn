from locale import D_FMT
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions


def custom_decision_region_plot(X, y, model, title):
    # Specify keyword arguments to be passed to underlying plotting functions
    scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
    contourf_kwargs = {'alpha': 0.2}
    scatter_highlight_kwargs = {'s': 60, 'label': 'Test data', 'alpha': 0.7, 'c': 'red'}

    # Plotting decision regions       
    plot_decision_regions(X=X, 
                          y=y,
                          clf=model, 
                          legend=2, 
                          scatter_kwargs=scatter_kwargs,
                          contourf_kwargs=contourf_kwargs,
                          scatter_highlight_kwargs=scatter_highlight_kwargs)
    plt.title(title)
    plt.show()
    
    
def box_plot(df, sampling_method):
    df_plot = df[df.sampling_method == sampling_method]
    sns.set(rc={'figure.figsize':(15,10)})
    sns.set_theme(style="ticks", palette="pastel")
    sns.boxplot(data=df_plot, x="sampling", y="performance gap", hue="model")
    plt.axhline(0, ls='--', c='red')
    plt.show()
    
    
def box_plot(df, y):
    sns.set(rc={'figure.figsize':(20,10)})
    sns.set_theme(style="ticks", palette="pastel")
    sns.boxplot(data=df, x="k", y=y, hue="metric")
    plt.show()
    
def annotate(data, regressor='kdn'):
    slope, intercept, r, p, std_err = st.linregress(data[regressor],data['score'])
    ax = plt.gca()
    plt.text(.05, .85, f'r={r:.2f}, p={p:.2g}',
            transform=ax.transAxes)
    plt.text(.05, .95, f'y={intercept:.2f} + {slope:.2f} ({std_err:.2f})x ',
            transform=ax.transAxes)
    plt.show()
    
    
def plot_reg(df, k, metric, score):
    df_corr = df[(df['k'] == k) & (df['metric'] == metric)]
    g = sns.lmplot(x=score, y='score', data=df_corr)
    annotate(df_corr, score)
    
    
def distribution_plot(df, title='Performance gap'):
    df_plot = df[3:22]
    x = df_plot.index

    plt.figure(figsize=(16, 6))

    plt.plot(x, df_plot['test_score_kdn'], 'o--', color='red', alpha=0.5, label='kdn          {:.3f} ({:.3f})'.format(df.loc['mean', 'test_score_kdn'],df.loc['std', 'test_score_kdn']))
    plt.plot(x, df_plot['test_score_dynamic_kdn'], 'o--', color='green', alpha=0.5, label='dkdn        {:.3f} ({:.3f})'.format(df.loc['mean', 'test_score_dynamic_kdn'],df.loc['std', 'test_score_dynamic_kdn']))
    plt.plot(x, df_plot['test_score_dynamic_kdn_full'], 'o--', color='blue', alpha=0.5, label='dkdn-full  {:.3f} ({:.3f})'.format(df.loc['mean', 'test_score_dynamic_kdn_full'],df.loc['std', 'test_score_dynamic_kdn_full']))

    plt.grid(axis='x', color='0.95')
    plt.legend()
    plt.title(title)

    plt.show()
    
    