from datetime import datetime
import sys
import os.path

# basic externals
import pandas as pd
import numpy as np
import logging
import feather

import warnings

from sklearn.externals import joblib
from argparse import ArgumentParser

# Imports the -file in parameter into a dataframe. Needs to be in the same format as original CSV from the project
def import_data(import_file):
    # Check if the file exists, otherwise get a new input
    while not import_file or not os.path.isfile(import_file):
        print("File not found", file=sys.stderr)
        import_file = input(
            "Enter a csv/xlsx file name containing customers first orders > "
        )

    # Importing our data from the file to a dataframe
    print("Importing", import_file, file=sys.stderr)
    file_path = import_file

    file_type = import_file.split(".")[1]
    if file_type == "xlsx":
        df = pd.read_excel(file_path)
    elif file_type == "csv":
        df = pd.read_csv(file_path)
    elif file_type == "feather":
        df = pd.read_feather(file_path)

    return df


# Transforming our DataFrame to fit our model
def transform(df):

    # Making sure orders have not been cancelled
    df_negative_quantity = df[df["Quantity"] < 0]
    if df_negative_quantity.shape[0] > 0:
        print(
            df_negative_quantity.shape[0],
            "items with negative quantity (cancels). Dropping",
            file=sys.stderr,
        )
        df = df.drop(df_negative_quantity.index)

    col = "InvoiceDate"
    df[col] = pd.to_datetime(df[col])
    df["Week"] = df["InvoiceDate"].apply(lambda x: xm_week(x))
    df["Time"] = df["InvoiceDate"].apply(lambda x: x.hour + x.minute * 0.016666)
    df["TotalPrice"] = df["UnitPrice"] * df["Quantity"]
    df["CountryUK"] = df["Country"].apply(lambda x: isCountry(x, "United Kingdom"))

    df_orders = df.groupby("InvoiceNo").agg(
        {
            "CustomerID": "first",
            "TotalPrice": "sum",
            "Quantity": "sum",
            "UnitPrice": "mean",
            "Week": "first",
            "StockCode": "count",
        }
    )

    # Flattening multi level index
    df_orders.columns = df_orders.columns.get_level_values(0)
    df_orders.columns = [
        "CustomerID",
        "TotalPrice",
        "Quantity",
        "UnitPrice_avg",
        "Week",
        "Distinct_items_count",
    ]

    if "FirstInvoice" not in df_orders.columns:
        df_first_order = (
            df_orders.reset_index()
            .groupby("CustomerID")
            .first()["InvoiceNo"]
            .rename("FirstInvoice")
        )
        df_orders = df_orders.join(df_first_order, on="CustomerID")

    df_first_orders = df_orders[df_orders["FirstInvoice"] == df_orders.index].copy()

    X = df_first_orders[
        [
            "TotalPrice",
            "Quantity",
            "UnitPrice_avg",
            "Week",
            "Distinct_items_count",
            "CustomerID",
        ]
    ]
    X.index = X.CustomerID
    X = X.drop("CustomerID", axis=1)
    return X


# Predicts our customers segment (from data X), with our previously generated model (reimported here using joblib)
def predict_segment(X):
    joblib_file = "exported_rf_model.joblib"
    model = joblib.load(joblib_file)

    results = model.predict(X)
    results = pd.DataFrame(
        results, index=np.array(X.index, dtype=int), columns=["Segment"]
    )
    print(results.shape[0], "customers found", file=sys.stderr)
    for i in range(1, 5):
        print(
            results[results.iloc[:, 0] == i].shape[0],
            "customers in segment",
            i,
            file=sys.stderr,
        )

    return results


# Sub function to turn date to week number
def xm_week(x):
    if x.year == 2010:
        return -52 + x.week
    else:
        return x.week


# Sub function to set country boolean
def isCountry(x, country):
    if x == country:
        return True
    else:
        return False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        dest="import_file",
        help="File to import data from. Can be csv, xlsx, feather.",
        metavar="FILE",
    )
    parser.add_argument(
        "-ex",
        "--export",
        dest="export_file",
        default="default",
        help="File to export the data into. Needs to be in csv.",
        metavar="FILE",
    )

    args = parser.parse_args()

    print(
        "This python app will categorize customers into predefined segments.",
        file=sys.stderr,
    )
    
    #To prevent deprecation error sklearn\ensemble\weight_boosting.py
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    # Importing via import function
    df = import_data(args.import_file)

    # Print dataframe shape
    print(
        df.shape[0],
        "rows and",
        df.shape[1],
        "columns found in the file.",
        file=sys.stderr,
    )

    # Transforming our data via the Transform funtion
    X = transform(df)

    # Apply our model to X to predict customers segments
    results = predict_segment(X)

    if args.export_file == "default":
        args.export_file = args.import_file.split(".")[0] + "_export.csv"

    results.to_csv(args.export_file)
    print("Results exported into", args.export_file, file=sys.stderr)
    # printing results on stdout, (the only print on stdout).
    if results.shape[0] < 100:
        print(results)
