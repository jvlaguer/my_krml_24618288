from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def plot_categ_feats(feat, df, target, th):
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax = ax.flatten()
    
    df_counts = df.groupby([feat, target]).size().reset_index(name='counts')
    df_pivot = df_counts.pivot_table(index=feat ,columns=target, aggfunc='sum', fill_value=0, margins=True)
    df_pcnt = df_pivot.div(df_pivot.iloc[:,-1], axis=0)
    
    value_filter = df_pcnt[('counts',1)] >= th
    df_pcnt_filtered = df_pcnt[value_filter].loc[:,('counts',1)].sort_values(ascending=False)
    sns.barplot(x=df_pcnt_filtered.index,y=df_pcnt_filtered.values, color='orange',ax=ax[0])
    
    df_cnts_filtered = df[feat].value_counts(ascending=False).filter(items = df_pcnt[value_filter].index).reindex(df_pcnt_filtered.index)
    sns.barplot(x=df_cnts_filtered.index,y=df_cnts_filtered.values, color='gray', ax=ax[1])
    
    ax[0].tick_params(axis='x', labelrotation=50)
    ax[1].tick_params(axis='x', labelrotation=50)
    
    ax[0].set_title('Pcnt of drafted per ' + feat)
    ax[1].set_title('Number of players per ' + feat)
    plt.show()

    return df_cnts_filtered