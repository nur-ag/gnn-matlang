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

print("Mutag: IGEL + GNNML3 vs NestedGNN")
print(ttest_ind_from_stats(
    92.50, 1.178511302, 10,
    87.9, 8.2, 10
))

print("Mutag: IGEL + GNNML3 vs GNN-AK")
print(ttest_ind_from_stats(
    92.50, 1.178511302, 10,
    91.7, 7.0, 10
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

print("Proteins: IGEL + GCN vs NestedGNN")
print(ttest_ind_from_stats(
    75.6666667, 0.346193, 10,
    74.2, 3.7, 10
))

print("Proteins: IGEL + GCN vs GNN-AK")
print(ttest_ind_from_stats(
    75.6666667, 0.346193, 10,
    77.1, 5.7, 10
))

print("PTC: IGEL + GAT vs k-hop")
print('--')

print("PTC: IGEL + GAT vs NestedGNN")
print('--')

print("PTC: IGEL + GAT vs GNN-AK")
print(ttest_ind_from_stats(
    66.29411765, 1.264667882, 10,
    67.7, 8.8, 10
))
