import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Loading train set
dataset = pd.read_csv("../house-prices/house-prices.csv")

# Note: the paramaters Order and PID are not taken into account in the analysis
# Nominal parameters:
nominal = ["MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood",
           "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st",
           "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", "Misc Feature",
           "Sale Type", "Sale Condition"]

# Continuous parameters:
continuous = ["Lot Frontage", "Lot Area", "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
             "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area", "Garage Area",
             "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Pool Area",
             "Misc Val"]

# Ordinal parameters:
ordinal = ["Lot Shape", "Utilities", "Land Slope", "Overall Qual", "Overall Cond", "Exter Qual", "Exter Cond",
           "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2", "Heating QC", "Electrical",
           "Kitchen Qual", "Functional", "Fireplace Qu", "Garage Finish", "Garage Qual", "Garage Cond",
           "Paved Drive", "Pool QC", "Fence"]

# Discrete parameters:
discrete = ["Year Built", "Year Remod/Add", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", "Half Bath",
            "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", "Fireplaces", "Garage Yr Blt", "Garage Cars",
            "Mo Sold", "Yr Sold"]

# Target parameter:
target = ["SalePrice"]

# Continuous variable:
# Compute Pearson coefficient for each continuous variable vs sale price:
output = np.asarray(dataset[target[0]])

pearson_coeff = {}
for var in continuous:
    # Extract the continuous parameter of interest
    continuous_parameters = np.asarray(dataset[var])

    # Get the values which are NaN
    nan_coeff = np.isnan(continuous_parameters)

    # Compute the pearson coeff without taking into account the nan value
    pearson_coeff[var] = pearsonr(continuous_parameters[~nan_coeff], output[~nan_coeff])[0]

print("The Pearson coefficient for each continuous parameters are:")
print(pearson_coeff)

# The relevant continuous parameters used for the regression task:
new_continuous = ["Gr Liv Area", "Garage Area", "Total Bsmt SF", "1st Flr SF", "Mas Vnr Area"]

# Compute Pearson coefficient for each discrete variable vs sale price:
output = np.asarray(dataset[target[0]])  # = sale price

# Discrete parameters
pearson_coeff = {}
for var in discrete:
    discrete_parameters = np.asarray(dataset[var])

    nan_coeff = np.isnan(discrete_parameters)

    pearson_coeff[var] = pearsonr(discrete_parameters[~nan_coeff], output[~nan_coeff])[0]

print("The Pearson coefficient for each discrete parameters are:")
print(pearson_coeff)

new_discrete = ["Year Built", "Year Remod/Add", "Full Bath", "TotRms AbvGrd", "Fireplaces"]