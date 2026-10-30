import pandas as pd
import numpy as np

df_global_property = pd.read_csv("shape_descriptors_cleaned.csv")



def data_standardization(data):
    # Calculate mean and standard deviation
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)

    # Standardize data
    standardized_data = (data - mean) / std_dev

    #normalized_data = (standardized_data - min(standardized_data)) / (max(standardized_data) - min(standardized_data))

    return standardized_data, mean, std_dev


volume_data = df_global_property["Volume"].to_numpy()
volume_standard, vol_mean, vol_std = data_standardization(volume_data)

surf_area_data = df_global_property["Surface Area"].to_numpy()
surf_area_standard, surf_area_mean, surf_area_std = data_standardization(surf_area_data)

compactness_data = df_global_property["Compactness"].to_numpy()
compactness_standard, comp_mean, comp_std = data_standardization(compactness_data)

rectangularity_data = df_global_property["Rectangularity"].to_numpy()
rectangularity_standard, rec_mean, rec_std = data_standardization(rectangularity_data)

diameter_data = df_global_property["Diameter"].to_numpy()
diameter_standard, dia_mean, dia_std = data_standardization(diameter_data)

convexity_data = df_global_property["Convexity"].to_numpy()
convexity_standard, cvx_mean, cvx_std = data_standardization(convexity_data)

eccentricity_data = df_global_property["Eccentricity"].to_numpy()
eccentricity_standard, ecc_mean, ecc_std = data_standardization(eccentricity_data)

standard_data = {
        "Class": df_global_property["Class"],
        "File": df_global_property["File"],
        "Volume": volume_standard,
        "Surface Area": surf_area_standard,
        "Compactness": compactness_standard,
        "Rectangularity": rectangularity_standard,
        "Diameter": diameter_standard,
        "Convexity": convexity_standard,
        "Eccentricity": eccentricity_standard
}

df_standard_data = pd.DataFrame(standard_data)
df_standard_data.to_csv("standardized_global_descriptors.csv", index=False)

mean_std = {
    "Volume Mean": [vol_mean],
    "Volume Std_Dev": [vol_std],
    "Surface Area Mean": [surf_area_mean],
    "Surface Area Std_Dev": [surf_area_std],
    "Compactness Mean": [comp_mean],
    "Compactness Std_Dev": [comp_std],
    "Rectangularity Mean": [rec_mean],
    "Rectangularity Std_Dev": [rec_std],
    "Diameter Mean": [dia_mean],
    "Diameter Std_Dev": [dia_std],
    "Convexity Mean": [cvx_mean],
    "Convexity Std_Dev": [cvx_std],
    "Eccentricity Mean": [ecc_mean],
    "Eccentricity Std_Dev": [ecc_std]
}

df_mean_std = pd.DataFrame(mean_std)
df_mean_std.to_csv("Mean_Std_Dev_Values.csv", index=False)