import pandas as pd

def transform_gold_data(df: pd.DataFrame) -> pd.DataFrame:
    
    gold = df.copy()
    
    gold.drop(['Vol.', 'Change %'], axis=1, inplace=True)
    NumCols = gold.columns.drop(['Date'])
    gold[NumCols] = gold[NumCols].replace({',': ''}, regex=True)
    gold[NumCols] = gold[NumCols].astype('float64')
    
    
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
        return df_cleaned

        # Remove outliers from the 'Price' column
    gold_cleaned = remove_outliers_iqr(gold, 'Price')
    gold = gold_cleaned.copy()
    return gold
