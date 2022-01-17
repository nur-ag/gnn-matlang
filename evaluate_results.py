import numpy as np
import pandas as pd

df = pd.read_csv('result_file.tsv', sep='\t')
df.columns = [col.lower() for col in df.columns]
df_no_zeros = df[(df.avg != 0.0) & (df.std != 0.0)]
agg_df = df_no_zeros.groupby(['script', 'distance', 'model']).agg({'avg': [np.mean, np.std]}).reset_index()
sort_df = agg_df.sort_values(['script', 'model', 'distance'])

dist_0_df = sort_df[sort_df.distance == 0]
dist_0_df.columns = ['script', 'distance', 'model', 'score_0_mean', 'score_0_std']
dist_1_df = sort_df[sort_df.distance == 1]
dist_1_df.columns = ['script', 'distance', 'model', 'score_1_mean', 'score_1_std']
dist_2_df = sort_df[sort_df.distance == 2]
dist_2_df.columns = ['script', 'distance', 'model', 'score_2_mean', 'score_2_std']

full_df = pd.merge(dist_0_df, dist_1_df, on=['script', 'model'])
full_df = pd.merge(full_df, dist_2_df, on=['script', 'model'])

full_df['dist_1_delta'] = full_df.score_1_mean - full_df.score_0_mean
full_df['dist_2_delta'] = full_df.score_2_mean - full_df.score_0_mean
full_df['dist_1_zscore'] = full_df.dist_1_delta / full_df.score_0_std
full_df['dist_2_zscore'] = full_df.dist_2_delta / full_df.score_0_std
full_df['is_dist_1_significative'] = np.abs(full_df.dist_1_zscore) < 1.96
full_df['is_dist_2_significative'] = np.abs(full_df.dist_2_zscore) < 1.96

script_deltas = full_df.groupby('script').mean()[['dist_1_delta', 'dist_2_delta']].reset_index()
script_deltas.columns = ['script', 'dist_1_delta', 'dist_2_delta']
print(script_deltas)

model_deltas = full_df.groupby('model').mean()[['dist_1_delta', 'dist_2_delta']].reset_index()
model_deltas.columns = ['model', 'dist_1_delta', 'dist_2_delta']
print(model_deltas)

script_model_deltas = full_df.groupby(['script', 'model']).mean()[['dist_1_delta', 'dist_2_delta']].reset_index()
script_model_deltas.columns = ['script', 'model', 'dist_1_delta', 'dist_2_delta']
print(script_model_deltas)

