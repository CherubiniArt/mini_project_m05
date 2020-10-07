from .py_scripts.run_toolchain import run_toolchain
import pkg_resources


def main():

    # All the parameters are selected
    continuous_parameters = ["Lot Frontage", "Lot Area", "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
                             "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area", "Garage Area",
                             "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Pool Area",
                             "Misc Val"]
    discrete_parameters = ["Year Built", "Year Remod/Add", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", "Half Bath",
                           "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", "Fireplaces", "Garage Yr Blt", "Garage Cars",
                           "Mo Sold", "Yr Sold"]
    ordinal_parameters = ["Lot Shape", "Utilities", "Land Slope", "Overall Qual", "Overall Cond", "Exter Qual",
                          "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2",
                          "Heating QC", "Electrical", "Kitchen Qual", "Functional", "Fireplace Qu", "Garage Finish",
                          "Garage Qual", "Garage Cond", "Paved Drive", "Pool QC", "Fence"]
    nominal_parameters = ["MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood",
                          "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl",
                          "Exterior 1st", "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air",
                          "Garage Type", "Misc Feature", "Sale Type", "Sale Condition"]

    # Path to the db used for the regression task
    db_path = "/house-prices/house-prices.csv"
    db_path = pkg_resources.resource_filename(__name__, db_path)

    # Protocol used to split the dataset into train/cv/test
    protocol = [0.8, 0.1, 0.1]

    # RF details:
    max_tree_depth_rf = 10
    n_trees = 50
    criterion = "mse"

    # Decision trees details:
    max_tree_depth_dt = 100

    save_fig = '/results/all_params_results.png'
    save_fig = pkg_resources.resource_filename(__name__, save_fig)
    seed = 42  # Used to fix the random_state of RF and decision tree regressors to ensure reproducibility

    run_toolchain(db_path, continuous_parameters, discrete_parameters, ordinal_parameters, nominal_parameters, protocol,
                  n_trees, criterion, seed, max_tree_depth_rf, max_tree_depth_dt, save_fig)

