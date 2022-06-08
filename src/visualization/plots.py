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
    
    
def box_plot_retraining(summary, feature, hline=0):
    sns.set(rc={'figure.figsize':(15,10)})
    sns.set_theme(style="ticks", palette="pastel")
    sns.boxplot(data=summary, x="proportion", y=feature, hue="model")
    plt.axhline(hline, ls='--', c='red')
    plt.show()