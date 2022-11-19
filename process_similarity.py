import os
import sys
import glob
import pandas as pd
from scipy.stats import ttest_ind_from_stats
from collections import deque


SIGNIFICANCE = 0.05


results = []
for file_name in glob.glob(f'*.txt'):
    with open(file_name, 'r') as f:
        last_line = deque(f, maxlen=1)[0]
        try:
            avg, std = list(map(float, last_line.split(' ')))
            results.append({
                'file': file_name,
                'exp_mean': avg,
                'exp_std': std
            })
        except ValueError as e:
            print(f'Could not parse results for {file_name}, skipping.')

results_df = pd.DataFrame(results)
results_df['experiment'] = [x.split('-')[0] for x in results_df.file]
results_df['seed'] = [x.split('-')[1] for x in results_df.file]
results_df['distance'] = [x.split('-')[2] for x in results_df.file]
results_df['vector_length'] = [x.split('-')[3] if x.split('-')[2] != '0' else '0' for x in results_df.file]
results_df['model'] = [x.split('-')[4] for x in results_df.file]
results_df['device'] = [x.split('-')[5] for x in results_df.file]
results_df.to_csv('raw_results.csv')

results_df = pd.read_csv('raw_results.csv')
exp_mean_df = results_df.groupby(['experiment', 'distance', 'vector_length', 'model', 'device']).agg({'exp_mean': ['mean', 'std', 'count']}).exp_mean.reset_index()
control_df = exp_mean_df[(exp_mean_df.distance == 0) & (exp_mean_df.vector_length == '0')].rename(columns={'mean': 'mean_control', 'std': 'std_control', 'count': 'count_control'}).drop(columns=['distance', 'vector_length'])
exp_control_df = pd.merge(exp_mean_df, control_df, on=['experiment', 'model', 'device'], how='inner')
exp_control_df['ttest'] = [
    ttest_ind_from_stats(row['mean'], row['std'], row['count'], row.mean_control, row.std_control, row.count_control) 
    for i, row in exp_control_df.iterrows()
]
exp_control_df['t_stat'] = [stat for (stat, pvalue) in exp_control_df['ttest']]
exp_control_df['p_value'] = [pvalue for (stat, pvalue) in exp_control_df['ttest']]
exp_control_df = exp_control_df.drop(columns=['ttest'])
exp_control_df.to_csv('experiment_stat_results.csv')

exp_control_df.loc[exp_control_df.experiment == 'exp_classify','t_stat':] *= -1

hypotheses = results_df.model.unique().size
sig_bonferroni = SIGNIFICANCE / hypotheses
best_sig_df = exp_control_df[exp_control_df.distance > 0].groupby(['experiment', 'model']).t_stat.min()
best_exp_control_df = exp_control_df.merge(best_sig_df, on=['experiment', 'model', 't_stat'], how='inner')
best_exp_control_df['is_significative_diff'] = best_exp_control_df.p_value < sig_bonferroni
best_exp_control_df.loc[best_exp_control_df.experiment == 'exp_classify','t_stat'] *= -1
best_exp_control_df = best_exp_control_df.sort_values(['experiment', 'model', 'distance', 'vector_length'])
best_exp_control_df.to_csv('best_results.csv')
