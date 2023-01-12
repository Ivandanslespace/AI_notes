data_path = "Data\Data.xlsx"
data_path_csv = "Data\\total_red.csv"

label = 'DR'
test_set = ['2009Q2', '2010Q4', '2012Q2', '2014Q2', '2016Q1']

scoring="neg_root_mean_squared_error"

TrainTestSplit_random_state = 110
train_size = 0.9

ploton = False
print_score = False


variables_complet = ['mean_1', 'median_1', 'p5_1', 'p10_1', 'p25_1', 'p75_1', 'p90_1', 'p95_1', 'mean_2', 'median_2', 'p5_2', 'p10_2',
       'p25_2', 'p75_2', 'p90_2', 'p95_2', 'mean_3', 'median_3', 'p5_3', 'p10_3', 'p25_3', 'p75_3', 'p90_3', 'p95_3', 'mean_4', 'median_4',
       'p5_4', 'p10_4', 'p25_4', 'p75_4', 'p90_4', 'p95_4', 'mean_5', 'median_5', 'p5_5', 'p10_5', 'p25_5', 'p75_5', 'p90_5', 'p95_5',
       'mean_6', 'median_6', 'p5_6', 'p10_6', 'p25_6', 'p75_6', 'p90_6', 'p95_6', 'mean_7', 'median_7', 'p5_7', 'p10_7', 'p25_7', 'p75_7',
       'p90_7', 'p95_7', 'mean_8', 'median_8', 'p5_8', 'p10_8', 'p25_8', 'p75_8', 'p90_8', 'p95_8', 'CD_TY_CLI_RCI_1', 'CD_TY_CLI_RCI_2',
       'CD_ETA_CIV_1', 'CD_ETA_CIV_2', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'CD_PROF_1', 'CD_PROF_2', 'CD_PROF_3', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2']

complet_hwage = ['mean_1', 'median_1', 'p5_1', 'p10_1', 'p25_1', 'p75_1', 'p90_1', 'p95_1', 'mean_2', 'median_2', 'p5_2', 'p10_2',
       'p25_2', 'p75_2', 'p90_2', 'p95_2', 'mean_3', 'median_3', 'p5_3', 'p10_3', 'p25_3', 'p75_3', 'p90_3', 'p95_3', 'mean_4', 'median_4',
       'p5_4', 'p10_4', 'p25_4', 'p75_4', 'p90_4', 'p95_4', 'mean_5', 'median_5', 'p5_5', 'p10_5', 'p25_5', 'p75_5', 'p90_5', 'p95_5',
       'mean_6', 'median_6', 'p5_6', 'p10_6', 'p25_6', 'p75_6', 'p90_6', 'p95_6', 'mean_7', 'median_7', 'p5_7', 'p10_7', 'p25_7', 'p75_7',
       'p90_7', 'p95_7', 'mean_8', 'median_8', 'p5_8', 'p10_8', 'p25_8', 'p75_8', 'p90_8', 'p95_8', 'CD_TY_CLI_RCI_1', 'CD_TY_CLI_RCI_2',
       'CD_ETA_CIV_1', 'CD_ETA_CIV_2', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'CD_PROF_1', 'CD_PROF_2', 'CD_PROF_3', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'h_wage', 'CAC40']

variables_10_50_90_optional = ['mean_1', 'median_1', 'p10_1', 'p90_1', 'mean_2', 'median_2', 'p10_2',
       'p90_2', 'mean_3', 'median_3', 'p10_3', 'p90_3', 'mean_4', 'median_4',
       'p10_4', 'p90_4', 'mean_5', 'median_5', 'p10_5', 'p90_5',
       'mean_6', 'median_6', 'p10_6', 'p90_6', 'mean_7', 'median_7', 'p10_7',
       'p90_7', 'mean_8', 'median_8', 'p10_8', 'p90_8', 'CD_TY_CLI_RCI_1', 'CD_TY_CLI_RCI_2',
       'CD_ETA_CIV_1', 'CD_ETA_CIV_2', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'CD_PROF_1', 'CD_PROF_2', 'CD_PROF_3', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2']

variables_10_50_90_optional_hwage = ['mean_1', 'median_1', 'p10_1', 'p90_1', 'mean_2', 'median_2', 'p10_2',
       'p90_2', 'mean_3', 'median_3', 'p10_3', 'p90_3', 'mean_4', 'median_4',
       'p10_4', 'p90_4', 'mean_5', 'median_5', 'p10_5', 'p90_5',
       'mean_6', 'median_6', 'p10_6', 'p90_6', 'mean_7', 'median_7', 'p10_7',
       'p90_7', 'mean_8', 'median_8', 'p10_8', 'p90_8', 'CD_TY_CLI_RCI_1', 'CD_TY_CLI_RCI_2',
       'CD_ETA_CIV_1', 'CD_ETA_CIV_2', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'CD_PROF_1', 'CD_PROF_2', 'CD_PROF_3', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'h_wage', 'CAC40']

variables_10_50_90 = ['mean_1', 'median_1', 'p10_1', 'p90_1', 'mean_2', 'median_2', 'p10_2',
       'p90_2', 'mean_3', 'median_3', 'p10_3', 'p90_3', 'mean_4', 'median_4',
       'p10_4', 'p90_4', 'mean_5', 'median_5', 'p10_5', 'p90_5',
       'mean_6', 'median_6', 'p10_6', 'p90_6', 'mean_7', 'median_7', 'p10_7',
       'p90_7', 'mean_8', 'median_8', 'p10_8', 'p90_8']

variables_10_50_90_hwage = ['mean_1', 'median_1', 'p10_1', 'p90_1', 'mean_2', 'median_2', 'p10_2',
       'p90_2', 'mean_3', 'median_3', 'p10_3', 'p90_3', 'mean_4', 'median_4',
       'p10_4', 'p90_4', 'mean_5', 'median_5', 'p10_5', 'p90_5',
       'mean_6', 'median_6', 'p10_6', 'p90_6', 'mean_7', 'median_7', 'p10_7',
       'p90_7', 'mean_8', 'median_8', 'p10_8', 'p90_8', 'h_wage', 'CAC40']

variables_pca_complet = [
    ['mean_1', 'median_1', 'p5_1', 'p10_1', 'p25_1', 'p75_1', 'p90_1', 'p95_1'],
    ['mean_2', 'median_2', 'p5_2', 'p10_2', 'p25_2', 'p75_2', 'p90_2', 'p95_2'],
    ['mean_3', 'median_3', 'p5_3', 'p10_3', 'p25_3', 'p75_3', 'p90_3', 'p95_3'],
    ['mean_4', 'median_4', 'p5_4', 'p10_4', 'p25_4', 'p75_4', 'p90_4', 'p95_4'],
    ['mean_5', 'median_5', 'p5_5', 'p10_5', 'p25_5', 'p75_5', 'p90_5', 'p95_5'],
    ['mean_6', 'median_6', 'p5_6', 'p10_6', 'p25_6', 'p75_6', 'p90_6', 'p95_6'],
    ['mean_7', 'median_7', 'p5_7', 'p10_7', 'p25_7', 'p75_7', 'p90_7', 'p95_7'],
    ['mean_8', 'median_8', 'p5_8', 'p10_8', 'p25_8', 'p75_8', 'p90_8', 'p95_8']
]
pca_complement_data = ['CD_TY_CLI_RCI_1', 'CD_ETA_CIV_1', 'CD_MOD_HABI_1', 'CD_PROF_1', 'CD_PROF_2', 'CD_PROF_3', 'CD_QUAL_VEH_1']
pca_complement_data_hw = ['CD_TY_CLI_RCI_1', 'CD_ETA_CIV_1', 'CD_MOD_HABI_1', 'CD_PROF_1', 'CD_PROF_2', 'CD_PROF_3', 'CD_QUAL_VEH_1', 'h_wage', 'CAC40']


# Variables_mRMR = [[['median_5'], ['mean_6'], ['p90_3']], [['median_5', 'p10_8'], ['mean_6', 'p90_6'], ['p90_3', 'mean_6']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1'], ['mean_6', 'p90_6', 'p75_6'], ['p90_3', 'mean_6', 'p95_3']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2'], ['mean_6', 'p90_6', 'p75_6', 'p10_1'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8', 'p10_4'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5', 'CD_MOD_HABI_1'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7', 'p90_1']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8', 'p10_4', 'CD_PROF_1'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7', 'p90_1', 'p90_2']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8', 'p10_4', 'CD_PROF_1', 'PIB'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'Tx_cho'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7', 'p90_1', 'p90_2', 'median_2']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8', 'p10_4', 'CD_PROF_1', 'PIB', 'CD_ETA_CIV_2'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'Tx_cho', 'p75_1'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7', 'p90_1', 'p90_2', 'median_2', 'mean_8']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8', 'p10_4', 'CD_PROF_1', 'PIB', 'CD_ETA_CIV_2', 'Inflation'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'Tx_cho', 'p75_1', 'CD_PROF_3'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7', 'p90_1', 'p90_2', 'median_2', 'mean_8', 'p75_2']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8', 'p10_4', 'CD_PROF_1', 'PIB', 'CD_ETA_CIV_2', 'Inflation', 'CD_ETA_CIV_1'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'Tx_cho', 'p75_1', 'CD_PROF_3', 'p5_3'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7', 'p90_1', 'p90_2', 'median_2', 'mean_8', 'p75_2', 'p25_7']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8', 'p10_4', 'CD_PROF_1', 'PIB', 'CD_ETA_CIV_2', 'Inflation', 'CD_ETA_CIV_1', 'p90_1'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'Tx_cho', 'p75_1', 'CD_PROF_3', 'p5_3', 'Inflation'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7', 'p90_1', 'p90_2', 'median_2', 'mean_8', 'p75_2', 'p25_7', 'p90_7']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8', 'p10_4', 'CD_PROF_1', 'PIB', 'CD_ETA_CIV_2', 'Inflation', 'CD_ETA_CIV_1', 'p90_1', 'p75_5'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'Tx_cho', 'p75_1', 'CD_PROF_3', 'p5_3', 'Inflation', 'p95_7'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7', 'p90_1', 'p90_2', 'median_2', 'mean_8', 'p75_2', 'p25_7', 'p90_7', 'p25_4']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8', 'p10_4', 'CD_PROF_1', 'PIB', 'CD_ETA_CIV_2', 'Inflation', 'CD_ETA_CIV_1', 'p90_1', 'p75_5', 'p25_5'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'Tx_cho', 'p75_1', 'CD_PROF_3', 'p5_3', 'Inflation', 'p95_7', 'median_3'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7', 'p90_1', 'p90_2', 'median_2', 'mean_8', 'p75_2', 'p25_7', 'p90_7', 'p25_4', 'mean_3']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8', 'p10_4', 'CD_PROF_1', 'PIB', 'CD_ETA_CIV_2', 'Inflation', 'CD_ETA_CIV_1', 'p90_1', 'p75_5', 'p25_5', 'p10_7'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'Tx_cho', 'p75_1', 'CD_PROF_3', 'p5_3', 'Inflation', 'p95_7', 'median_3', 'p75_4'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7', 'p90_1', 'p90_2', 'median_2', 'mean_8', 'p75_2', 'p25_7', 'p90_7', 'p25_4', 'mean_3', 'CD_ETA_CIV_1']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8', 'p10_4', 'CD_PROF_1', 'PIB', 'CD_ETA_CIV_2', 'Inflation', 'CD_ETA_CIV_1', 'p90_1', 'p75_5', 'p25_5', 'p10_7', 'median_1'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'Tx_cho', 'p75_1', 'CD_PROF_3', 'p5_3', 'Inflation', 'p95_7', 'median_3', 'p75_4', 'p10_6'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7', 'p90_1', 'p90_2', 'median_2', 'mean_8', 'p75_2', 'p25_7', 'p90_7', 'p25_4', 'mean_3', 'CD_ETA_CIV_1', 'CD_ETA_CIV_2']], [['median_5', 'p10_8', 'CD_QUAL_VEH_1', 'CD_QUAL_VEH_2', 'p10_1', 'mean_8', 'mean_5', 'p5_4', 'p25_8', 'p25_1', 'p95_8', 'p90_8', 'median_8', 'p10_4', 'CD_PROF_1', 'PIB', 'CD_ETA_CIV_2', 'Inflation', 'CD_ETA_CIV_1', 'p90_1', 'p75_5', 'p25_5', 'p10_7', 'median_1', 'CD_MOD_HABI_2'], ['mean_6', 'p90_6', 'p75_6', 'p10_1', 'p5_1', 'p95_6', 'p25_6', 'mean_1', 'p25_1', 'median_6', 'p25_5', 'median_1', 'p10_5', 'CD_MOD_HABI_1', 'CD_MOD_HABI_2', 'Tx_cho', 'p75_1', 'CD_PROF_3', 'p5_3', 'Inflation', 'p95_7', 'median_3', 'p75_4', 'p10_6', 'p75_5'], ['p90_3', 'mean_6', 'p95_3', 'CD_MOD_HABI_2', 'PIB', 'CD_MOD_HABI_1', 'p95_2', 'p90_6', 'mean_2', 'p95_6', 'mean_7', 'mean_1', 'p75_7', 'p90_1', 'p90_2', 'median_2', 'mean_8', 'p75_2', 'p25_7', 'p90_7', 'p25_4', 'mean_3', 'CD_ETA_CIV_1', 'CD_ETA_CIV_2', 'p75_4']]]