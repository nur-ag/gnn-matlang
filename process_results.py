import os
import sys
import glob
import pandas as pd
from scipy.stats import ttest_ind_from_stats
from collections import deque


SIGNIFICANCE = 0.05


def fetch_last_lines(file_path, n=4):
    with open(file_path, 'r') as f:
        return list(deque(f, maxlen=n))


def check_lines_pattern(last_lines):
    if len(last_lines) != 4:
        return False
    # Last epoch of the last split
    first = last_lines[0]
    remaining = last_lines[1:]
    if not first.startswith('09 Epoch:'):
        return False
    # This script doesn't print final epoch scores!
    second = remaining[0]
    if second.startswith('09 Epoch:'):
        remaining = remaining[1:]
    # Two or three 'score' lines -- for the epoch and complete
    for line in remaining:
        try:
            value = float(line.strip())
        except ValueError:
            return False
    return True


def file_to_dict(file_name):
    with open(file_name, 'r') as f:
        lines = f.read().strip().splitlines()
        mean, std = lines[-2:]
    return {
        "file": file_name[:-4], 
        "exp_mean": float(mean), 
        "exp_std": float(std)
    }


results = []
for file_name in glob.glob(f'*.txt'):
    last_lines = fetch_last_lines(file_name)
    if check_lines_pattern(last_lines):
        experiment_tuple = file_to_dict(file_name)
        results.append(experiment_tuple)


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

hypotheses = results_df.model.unique().size
sig_bonferroni = SIGNIFICANCE / hypotheses
best_sig_df = exp_control_df[exp_control_df.distance > 0].groupby(['experiment', 'model']).t_stat.max()
best_exp_control_df = exp_control_df.merge(best_sig_df, on=['experiment', 'model', 't_stat'], how='inner')
best_exp_control_df['is_significative_diff'] = best_exp_control_df.p_value < sig_bonferroni
best_exp_control_df = best_exp_control_df.sort_values(['experiment', 'model', 'distance', 'vector_length'])
best_exp_control_df.to_csv('best_results.csv')
