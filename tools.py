import numpy as np
import pandas as pd
import streamlit as st
from lightweight_mmm import media_transforms


def format_title_media(title: str) -> str:
    if title in ["TV", "OOH"]:
        return title
    else:
        return title[0] + title[1:].lower()


def load_excel(file: str) -> dict:
    """
    Load an Excel file into a dictionary of DataFrames, one for each sheet.

    Parameters:
    file: str or file-like object
        The path or file-like object for the Excel file.

    Returns:
    dict
        A dictionary where each key is a sheet name, and the corresponding value is a DataFrame.
    """
    excel_data = pd.read_excel(file, sheet_name=None)
    return excel_data


def format_currency(amount) -> str:
    """
    Format a numeric amount into a currency string with spaces as thousand separators and euros as the currency.

    Parameters:
    amount: float or int
        The numeric amount to format.

    Returns:
    str
        The formatted amount as a string in the format 'X XXX €'.
    """
    return "{:,.0f} €".format(amount).replace(",", " ")


def filter_data(df: pd.DataFrame, dates: list) -> pd.DataFrame:
    """
    Filter a DataFrame to include only rows within a specified date range.

    Parameters:
    df: pd.DataFrame
        The DataFrame containing a 'Date' column.
    dates: list
        A list containing two date values [start_date, end_date] used for filtering.

    Returns:
    pd.DataFrame
        A filtered DataFrame with dates within the specified range.
    """
    new_df = df.copy()
    new_df["Date"] = pd.to_datetime(new_df["Date"], dayfirst=True)
    new_df = new_df[new_df["Date"] > dates[0]]
    new_df = new_df[new_df["Date"] < dates[1]]
    return new_df


def preprocess_data(df: pd.DataFrame, dates: list) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by filtering, selecting media-related columns, and aggregating data by media type and channel.

    Parameters:
    df: pd.DataFrame
        The input DataFrame containing media data.
    dates: list
        A list containing two date values [start_date, end_date] for filtering the data.

    Returns:
    pd.DataFrame
        A DataFrame grouped by 'Media Type', 'Media Channel', and total 'Amount' spent.
    """
    new_df = df.copy()
    new_df.drop(["index"], axis=1, inplace=True)
    to_keep = [c for c in new_df.columns if "Media" in c] + ["Date"]
    new_df = new_df[to_keep]
    new_df = filter_data(new_df, dates)
    grouped_df = (
        (new_df.drop("Date", axis=1).sum())
        .T.reset_index(name="Amount")
        .rename({"index": "Media"}, axis=1)
    )
    grouped_df["Media Type"] = [v.split("_")[1] for v in grouped_df["Media"]]
    grouped_df["Media Channel"] = [v.split("_")[2] for v in grouped_df["Media"]]
    grouped_df.drop("Media", axis=1, inplace=True)
    return grouped_df[["Media Type", "Media Channel", "Amount"]]


def preprocess_budget(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the budget data by reshaping and calculating allocations for previous and optimized budgets.

    Parameters:
    df: pd.DataFrame
        The input DataFrame with media budget information.

    Returns:
    tuple of pd.DataFrame
        Two DataFrames: one for previous budget and one for optimized budget, both with calculated allocations.
    """
    new_df = df.reset_index().melt(
        id_vars="index",
        value_vars=["previous_budget", "optimized_budget"],
        var_name="Budget",
        value_name="Amount",
    )
    new_df.insert(0, "Media Type", [w.split("_")[1] for w in new_df["index"]])
    new_df.rename({"index": "Media Channel"}, axis=1, inplace=True)
    new_df["Media Channel"] = [c.split("_")[-1] for c in new_df["Media Channel"]]

    previous = new_df[new_df["Budget"] == "previous_budget"].reset_index(drop=True)
    previous["Allocation"] = previous["Amount"] / previous["Amount"].sum()

    optimized = new_df[new_df["Budget"] == "optimized_budget"].reset_index(drop=True)
    optimized["Allocation"] = optimized["Amount"] / optimized["Amount"].sum()
    return previous.drop("Budget", axis=1), optimized.drop("Budget", axis=1)


def preprocess_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the ROI data by extracting relevant columns and removing unnecessary ones.

    Parameters:
    df: pd.DataFrame
        The input DataFrame containing ROI data.

    Returns:
    pd.DataFrame
        A DataFrame with relevant columns for ROI analysis.
    """
    new_df = df.copy()
    new_df.insert(0, "Media Type", [w.split("_")[1] for w in new_df["index"]])
    new_df.rename({"index": "Media Channel"}, axis=1, inplace=True)
    new_df["Media Channel"] = [c.split("_")[-1] for c in new_df["Media Channel"]]
    new_df.drop(
        [
            "contribution",
            "confidence_int_inf_contribution",
            "confidence_int_sup_contribution",
        ],
        axis=1,
        inplace=True,
    )
    return new_df


def get_budget(amount: float, breakdown: pd.DataFrame) -> pd.DataFrame:
    """
    Distribute a given budget across media channels based on their current allocations.

    Parameters:
    amount: float
        The total new budget to allocate.
    breakdown: pd.DataFrame
        The DataFrame containing media channels and their current amounts.

    Returns:
    pd.DataFrame
        A DataFrame with the new budget distributed across channels and updated allocations.
    """
    new_budget = breakdown.copy()
    allocations = new_budget["Amount"] / new_budget["Amount"].sum()
    new_budget["New Amount"] = allocations * amount
    new_budget["Allocation"] = allocations * 100
    return new_budget


def access_type_budget(
    df: pd.DataFrame, media_type: str, val_type: str
) -> pd.DataFrame:
    """
    Access the budget value for a specific media type and value type (e.g., amount or allocation).

    Parameters:
    df: pd.DataFrame
        The DataFrame containing budget information.
    media_type: str
        The media type to filter.
    val_type: str
        The value type to return (e.g., "Amount", "Allocation").

    Returns:
    float
        The budget value for the specified media type and value type.
    """
    return df.loc[df["Media Type"] == media_type, [val_type]].values[0][0]


def access_channel_budget(
    df: pd.DataFrame, media_channel: str, val_type: str
) -> pd.DataFrame:
    """
    Access the budget value for a specific media channel and value type (e.g., amount or allocation).

    Parameters:
    df: pd.DataFrame
        The DataFrame containing budget information.
    media_channel: str
        The media channel to filter.
    val_type: str
        The value type to return (e.g., "Amount", "Allocation").

    Returns:
    float
        The budget value for the specified media channel and value type.
    """
    return df.loc[df["Media Channel"] == media_channel, [val_type]].values[0][0]


def get_channel_allocation(
    df: pd.DataFrame, amount: float, media_type: str
) -> pd.DataFrame:
    """
    Allocate a budget to media channels based on their current allocations for a specific media type.

    Parameters:
    df: pd.DataFrame
        The DataFrame containing media channel data.
    amount: float
        The total budget to allocate.
    media_type: str
        The media type for which the allocation is calculated.

    Returns:
    pd.DataFrame
        A DataFrame with the new budget distributed across channels for the specified media type.
    """
    new_budget = df.copy()
    new_budget = new_budget[new_budget["Media Type"] == media_type]
    allocations = new_budget["Amount"] / new_budget["Amount"].sum()
    new_budget["New Amount"] = allocations * amount
    new_budget["Allocation"] = allocations * 100
    return new_budget


def saveTypeBudget(
    df: pd.DataFrame, values: dict[str, float], usePerc: bool, budget: float
) -> pd.DataFrame:
    """
    Update a DataFrame's budget allocation based on either percentage or absolute values.

    Parameters:
    df: pd.DataFrame
        The DataFrame containing media budget data.
    values: dict[str, float]
        A dictionary where keys are media channels and values are either percentage or absolute budget values.
    usePerc: bool
        If True, values are interpreted as percentages; otherwise, they are interpreted as absolute amounts.
    budget: float
        The total budget for allocation.

    Returns:
    pd.DataFrame
        A DataFrame with updated 'Allocation' and 'New Amount' columns.
    """
    if usePerc:
        for k, v in values.items():
            df.loc[df["Media Channel"] == k, ["Allocation"]] = v
            df["New Amount"] = budget * df["Allocation"] / 100
    else:
        for k, v in values.items():
            df.loc[df["Media Channel"] == k, ["New Amount"]] = v
            df["Allocation"] = df["New Amount"] / budget * 100
    return df


def compare_global_roi(dict_of_df: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compare the ROI across different scenarios by aggregating contribution and investment for each DataFrame.

    Parameters:
    dict_of_df: dict[str, pd.DataFrame]
        A dictionary where keys are scenario names and values are DataFrames containing 'Contribution' and 'Investment'.

    Returns:
    pd.DataFrame
        A DataFrame with ROI values for each scenario.
    """
    roi_per_df = {
        k: [sum(v["Contribution"]) / sum(v["Investment"])]
        for k, v in dict_of_df.items()
    }
    compare_df = pd.DataFrame(roi_per_df).T.reset_index()
    compare_df.columns = ["Scenario", "ROI"]
    compare_df["Values"] = compare_df["ROI"].apply(lambda x: f"{x:.2f}")
    return compare_df


def makeSimulation(
    budget_allocation: pd.DataFrame, target_scaler, media_scaler, model
) -> pd.DataFrame:
    """
    Simulate the effect of budget allocations on media contribution using a pre-trained model.

    Parameters:
    budget_allocation: pd.DataFrame
        The DataFrame containing media types, channels, and budget allocations.
    target_scaler: scaler object
        A scaler to inverse-transform the simulation results.
    media_scaler: scaler object
        A scaler used to transform the input data for the model.
    model: trained model object
        The pre-trained model used for simulation.

    Returns:
    pd.DataFrame
        A DataFrame with media contributions based on the simulation.
    """
    simulationData = budget_allocation.copy()
    simulationData["Media_var"] = [
        (
            "Media_" + c["Media Type"] + "_" + c["Media Channel"] + "_€"
            if c["Media Channel"] not in ["Search", "Social", "Display"]
            else "Media_" + c["Media Type"] + "_€"
        )
        for c in budget_allocation.to_dict(orient="records")
    ]
    simulationRecords = (
        simulationData[["Media_var", "Amount"]]
        .set_index("Media_var")["Amount"]
        .to_dict()
    )

    # Generate test data for simulation
    data_to_test = pd.DataFrame(
        np.ones((10, len(simulationData["Media_var"]))) / 10,
        columns=list(simulationRecords.keys()),
    )
    data_to_test = data_to_test / data_to_test.sum(axis=0)

    # Scale test data by simulation records
    for c in data_to_test.columns:
        data_to_test[c] = data_to_test[c] * simulationRecords[c]

    data_to_test = data_to_test.to_numpy()

    # Apply carryover effects and transformations
    carryover_list = [
        media_transforms.carryover(
            media_scaler.transform(data_to_test),
            model.trace["ad_effect_retention_rate"][i],
            model.trace["peak_effect_delay"][i],
        )
        for i in range(model.trace["ad_effect_retention_rate"].shape[0])
    ]
    transformed_data_list = [
        media_transforms.apply_exponent_safe(
            data=carryover_list[i], exponent=model.trace["exponent"][i]
        )
        for i in range(model.trace["exponent"].shape[0])
    ]
    contribution_list = [
        model.trace["coef_media"][i] * transformed_data_list[i]
        for i in range(len(model.trace["coef_media"]))
    ]

    # Calculate contribution and merge with original data
    contributionSimulated = target_scaler.inverse_transform(
        np.array(contribution_list).mean(axis=0)
    )
    contributionSimulated = (
        pd.DataFrame(contributionSimulated, columns=list(simulationRecords.keys()))
        .fillna(0)
        .sum()
        .reset_index(name="Contribution")
        .rename({"index": "Media_var"}, axis=1)
    )
    contributionSimulated = simulationData.merge(
        contributionSimulated, on="Media_var", how="left"
    )
    return contributionSimulated


def checkGlobalPred(
    budget_allocation: pd.DataFrame, target_scaler, media_scaler, model
) -> float:
    """
    Check the global prediction of a media budget allocation scenario.

    Parameters:
    budget_allocation: pd.DataFrame
        The DataFrame containing media types, channels, and budget allocations.
    target_scaler: scaler object
        A scaler to inverse-transform the simulation results.
    media_scaler: scaler object
        A scaler used to transform the input data for the model.
    model: trained model object
        The pre-trained model used for predictions.

    Returns:
    float
        The sum of global predictions for the provided budget allocation.
    """
    simulationData = budget_allocation.copy()
    simulationData["Media_var"] = [
        (
            "Media_" + c["Media Type"] + "_" + c["Media Channel"] + "_€"
            if c["Media Channel"] not in ["Search", "Social", "Display"]
            else "Media_" + c["Media Type"] + "_€"
        )
        for c in budget_allocation.to_dict(orient="records")
    ]
    simulationRecords = (
        simulationData[["Media_var", "Amount"]]
        .set_index("Media_var")["Amount"]
        .to_dict()
    )

    # Generate test data for simulation
    data_to_test = pd.DataFrame(
        np.ones((10, len(simulationData["Media_var"]))) / 10,
        columns=list(simulationRecords.keys()),
    )

    # Scale test data by simulation records
    for c in data_to_test.columns:
        data_to_test[c] = data_to_test[c] * simulationRecords[c]

    data_to_test = data_to_test.to_numpy()

    # Transform and predict
    transformed_data = media_scaler.transform(data_to_test)
    raw_preds = model.predict(transformed_data, target_scaler=target_scaler)
    return raw_preds.mean(axis=0).sum()


def get_months(df: pd.DataFrame) -> list:
    """
    Extract unique months from a DataFrame with a date column.

    Parameters:
    df: pd.DataFrame
        The DataFrame containing a 'Date' column.

    Returns:
    list
        A list of unique months as datetime.date objects.
    """
    new_df = df.copy()
    new_df["Date"] = pd.to_datetime(new_df["Date"], format="%d/%m/%Y")
    unique_months = new_df["Date"].dt.to_period("M").unique()
    unique_months = pd.to_datetime(unique_months.to_timestamp()).date
    return list(unique_months)


def load_css(file_path: str):
    """
    Load and apply custom CSS to the Streamlit app.

    Parameters:
    file_path: str
        The path to the CSS file.
    """
    with open(file_path, "r") as file:
        css = file.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
