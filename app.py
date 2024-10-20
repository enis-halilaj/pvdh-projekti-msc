import argparse
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def load_data(filepath):
    return pd.read_csv(filepath)

def show_data_types(df):
    data_types = df.dtypes
    return data_types

def fill_missing_values(df):
    imputer = SimpleImputer(strategy='median')
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_filled

def reduce_dimensions(df):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df.select_dtypes(include=['number']))
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    return pd.concat([df.drop(df.select_dtypes(['number']).columns, axis=1), df_pca], axis=1)

def check_duplicates(df):
    duplicates = df[df.duplicated(keep=False)]
    return duplicates

def show_missing_values(df):
    missing_values = df.isnull().sum()
    return missing_values

def main(args):
    df = load_data(args.filepath)
    if args.fill_missing:
        df = fill_missing_values(df)
    if args.show_types:
        data_types = show_data_types(df)
        print("Llojet e të dhënave për secilën kolonë:")
        print(data_types)
    if args.reduce_dim:
        df = reduce_dimensions(df)
    if args.check_duplicates:
        dupes = check_duplicates(df)
        print("Rreshtat duplikat:")
        print(dupes)
    if args.show_missing:
        missing_values = show_missing_values(df)
        print("Mungesa e vlerave për kolonë:")
        print(missing_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Application")
    parser.add_argument('filepath', type=str, help="Path to the data file")
    parser.add_argument('--fill_missing', action='store_true', help="Fill missing values")
    parser.add_argument('--reduce_dim', action='store_true', help="Reduce dimensions")
    parser.add_argument('--check_duplicates', action='store_true', help="Check for duplicate rows")
    parser.add_argument('--show_missing', action='store_true', help="Show missing values per column")
    parser.add_argument('--show_types', action='store_true', help="Show data types of columns")

    args = parser.parse_args()
    main(args)