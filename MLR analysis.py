import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def multiple_linear_regression_analysis(file_path):
    # Read all sheets from the Excel file
    excel_data = pd.ExcelFile(file_path)
    sheet_names = excel_data.sheet_names

    all_results = {}
    all_scaled_data = {}  # Dictionary to store scaled data
    residuals_data = {}   # Dictionary to store residuals

    for sheet in sheet_names:
        # Load the sheet into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet)

        # Drop 'Site ID' column if present
        if 'Site ID' in df.columns:
            df = df.drop(columns=['Site ID'])

        # Define independent and dependent variables
        y = df['Total toxicity']
        x = df.drop(columns=['Total toxicity'])

        # Handle non-numeric values in X
        for column in x.columns:
            if x[column].dtype == object:
                x[column] = x[column].replace('n.d.', 0).astype(float)

        # Standardize the independent variables
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Convert scaled data back to DataFrame, replace NaNs with 0
        x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns).fillna(0)

        # Save scaled data to the dictionary
        all_scaled_data[sheet] = x_scaled_df

        # Fit the multiple linear regression model without adding a constant
        model = sm.OLS(y, x_scaled_df).fit()

        # Calculate R-squared and adjusted R-squared
        r_squared = model.rsquared
        adjusted_r_squared = model.rsquared_adj

        # Get the standardized coefficients and p-values
        coefficients = model.params
        p_values = model.pvalues

        # Calculate the contribution percentage of each coefficient
        coef_sum = np.sum(np.abs(coefficients))
        percentage_contributions = np.abs(coefficients) / coef_sum * 100

        # Calculate residuals
        residuals = model.resid

        # Save residuals to the dictionary
        residuals_data[sheet] = pd.DataFrame({'residuals': residuals})

        # Create a DataFrame to display results including R-squared
        results = pd.DataFrame({
            'pollutant': x.columns,
            'standardized_coefficients': coefficients,
            'percentage_contribution': percentage_contributions,
            'p_value': p_values,
            'R_squared': [r_squared] * len(x.columns),
            'Adjusted_R_squared': [adjusted_r_squared] * len(x.columns)
        })


        all_results[sheet] = results

    return all_results, all_scaled_data, residuals_data

# Perform the analysis for the provided file path
input_file_path = 'MLR analysis.xlsx'
results_dict, scaled_data_dict, residuals_dict = multiple_linear_regression_analysis(input_file_path)

# Export the results to a new Excel file
output_file_path = 'MLR_analysis_results.xlsx'
scaled_data_path = 'Standardized_data.xlsx'
residuals_path = 'Residuals_data.xlsx'
with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
    for sheet_name, result_df in results_dict.items():
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)

with pd.ExcelWriter(scaled_data_path, engine='openpyxl') as writer:
    for sheet_name, data_df in scaled_data_dict.items():
        data_df.to_excel(writer, sheet_name=sheet_name, index=False)

with pd.ExcelWriter(residuals_path, engine='openpyxl') as writer:
    for sheet_name, residuals_df in residuals_dict.items():
        residuals_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Analysis results have been exported to {output_file_path}")
print(f"Standardized data has been exported to {scaled_data_path}")
print(f"Residuals data has been exported to {residuals_path}")

