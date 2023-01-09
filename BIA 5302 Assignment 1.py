#Copy and Paste Python Code into Jupyter Notebook. Ensure notebook is in the same folder as "BostonHousing.xls" before executing any code.

import pandas as pd
import numpy as np
import sklearn as sk
import math

boston_housing_df = pd.read_excel("BostonHousing.xls", sheet_name = "Data") #Loads the data into the variable "boston_housing_df"

boston_housing_df.head() #Displays top 5 rows of the dataframe

boston_housing_df['CRIM'].unique() #each unique() line gets all unique/distinct values for specified column

boston_housing_df['ZN'].unique()

boston_housing_df['INDUS'].unique()

boston_housing_df['CHAS'].unique()

boston_housing_df['NOX'].unique()

boston_housing_df['RM'].unique()

boston_housing_df['AGE'].unique()

boston_housing_df['DIS'].unique()

boston_housing_df['RAD'].unique()

boston_housing_df['TAX'].unique()

boston_housing_df['PTRATIO'].unique()


def is_float(value: str) -> bool:
    '''Given a string value, checks if this value can be converted to float.
    If it can, it returns true. Otherwise, it returns false.
    '''
    
    status = False
    
    try:
        float(value)
        status = True
    
    except ValueError:
        status = False
    
    return status
   

def highlight_missing_value(value: str) -> str:
    '''Applies the is_float() function on value and determines whether that row should
    be highlighted as missing value or not.
    '''
    
    color = 'yellow' if is_float(value) == False or math.isnan(value) else ''
    return 'background-color: {}'.format(color)



boston_housing_df.style.applymap(highlight_missing_value, subset=['CRIM','ZN','INDUS','CHAS', 'NOX', 'RM','AGE','DIS','RAD','TAX']) 
#Applies a highlight function to highlight any missing value in yellow



def pt_ratio_outlier(value: str) -> str:
    '''Given a string value in the PTRATIO column, checks if the value is a string, a number greater than 55,
    a number between 55 and 25 or a number less than 10. The function labels non-numeric values as 'a', errors
    due to decimal places as 'b', genuine cases of outliers as 'c' or 'normal' otherwise.
    '''
    
    code = ''
    
    if not is_float(value):
        code = 'a'
    
    elif 25 < float(value) < 55:
        code = 'c'
    
    elif float(value) >= 55 or float(value) < 10:
        code = 'b'
    
    else:
        code = 'normal'
    
    return code   
    

def highlight_outlier(value: str) -> str:
    '''Applies the pt_ratio_outlier() function on value and determines whether that row should
    be highlighted as an outlier or not.
    '''
    
    color = 'yellow' if pt_ratio_outlier(value) != 'normal' else ''
    return 'background-color: {}'.format(color)
    

boston_housing_df['OUTLIER'] = boston_housing_df['PTRATIO'].apply(pt_ratio_outlier) # Creates new column 'OUTLIER' to indicate
# any outliers present in the preceeding 'PTRATIO' column in the row adjacent to it


boston_housing_df.style.applymap(highlight_outlier, subset=['PTRATIO']) # Applies a highlight function to highlight any PTRATIO
# outliers in yellow

# Part C:
boston_housing_df_NaN = boston_housing_df.copy()

boston_housing_df_NaN.drop('OUTLIER', axis=1, inplace=True)

boston_housing_df_NaN = boston_housing_df_Nan.apply(pd.to_numeric, errors='coerce')
# 1. Forces all values to convert to a numeric data type, if not numeric the value defaults into NaN

boston_housing_df_imputed = boston_housing_df_NaN.copy()

# 2A Omission
boston_housing_df_Omission = boston_housing_df_NaN.dropna()

boston_housing_df_Omission.iloc[0:20]

# 2B  Imputation
for column in boston_housing_df_imputed:
    column_median = boston_housing_df_imputed[column].median()
    boston_housing_df_imputed[column] = boston_housing_df_imputed[column].fillna(value=column_median)
    
boston_housing_df_imputed.iloc[35:50]

boston_housing_df.iloc[35:50]
