from scipy.stats import ttest_ind_from_stats

# Mutag: IGEL + GNNML3 vs GSN
print(ttest_ind_from_stats(
    92.50, 1.178511302, 10,
    92.2, 7.5, 10
))

# Proteins: IGEL + GCN vs ESAN
print(ttest_ind_from_stats(
    75.6666667, 0.346193, 10,
    76.7, 4.1, 10
))

# PTC: IGEL + GAT vs ESAN
print(ttest_ind_from_stats(
    66.29411765, 1.264667882, 10,
    69.2, 6.5, 10
))
