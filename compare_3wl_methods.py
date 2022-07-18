from scipy.stats import ttest_ind_from_stats

print("Mutag: IGEL + GNNML3 vs k-hop")
print(ttest_ind_from_stats(
    92.50, 1.178511302, 10,
    87.9, 1.2, 10
))

print("Mutag: IGEL + GNNML3 vs GSN")
print(ttest_ind_from_stats(
    92.50, 1.178511302, 10,
    92.2, 7.5, 10
))

print("Mutag: IGEL + GNNML3 vs ESAN")
print(ttest_ind_from_stats(
    92.50, 1.178511302, 10,
    91.1, 7.0, 10
))

print("Proteins: IGEL + GCN vs k-hop")
print(ttest_ind_from_stats(
    75.6666667, 0.346193, 10,
    75.3, 0.4, 10
))

print("Proteins: IGEL + GCN vs GSN")
print(ttest_ind_from_stats(
    75.6666667, 0.346193, 10,
    76.6, 5.0, 10
))

print("Proteins: IGEL + GCN vs ESAN")
print(ttest_ind_from_stats(
    75.6666667, 0.346193, 10,
    76.7, 4.1, 10
))

print("PTC: IGEL + GAT vs k-hop")
print('--')

print("PTC: IGEL + GAT vs GSN")
print(ttest_ind_from_stats(
    66.29411765, 1.264667882, 10,
    68.2, 7.2, 10
))

print("PTC: IGEL + GAT vs ESAN")
print(ttest_ind_from_stats(
    66.29411765, 1.264667882, 10,
    69.2, 6.5, 10
))
