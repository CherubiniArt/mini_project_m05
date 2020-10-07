from .py_scripts.run_toolchain import run_toolchain

# Only relevant params selected (see details on relevant_params_selection.py)
continuous_parameters = ["Gr Liv Area", "Garage Area", "Total Bsmt SF", "1st Flr SF", "Mas Vnr Area"]
discrete_parameters = ["Year Built", "Year Remod/Add", "Full Bath", "TotRms AbvGrd", "Fireplaces"]
ordinal_parameters = []
nominal_parameters = []

# Path to the db used for the regression task
db_path = "./house-prices/house-prices.csv"

# Protocol used to split the dataset into train/cv/test
protocol = [0.8, 0.1, 0.1]

# RF details:
max_tree_depth_rf = 10
n_trees = 50
criterion = "mse"

# Decision trees details:
max_tree_depth_dt = 100

save_fig = './results/relevant_params_results.png'
seed = 42  # Used to fix the random_state of RF and decision tree regressors to ensure reproducibility

run_toolchain(db_path, continuous_parameters, discrete_parameters, ordinal_parameters, nominal_parameters, protocol,
              n_trees, criterion, seed, max_tree_depth_rf, max_tree_depth_dt, save_fig)
