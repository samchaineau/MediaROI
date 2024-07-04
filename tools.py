import pandas as pd

def load_excel(file):
    excel_data = pd.read_excel(file, sheet_name=None)
    return excel_data

def preprocess_data(df : pd.DataFrame,
                    dates : list):
    new_df = df.copy()
    new_df.drop(["index"], axis = 1, inplace=True)
    to_keep = [c for c in new_df.columns if "Media" in c]+["Date"]
    new_df = new_df[to_keep]
    new_df["Date"] = pd.to_datetime(new_df["Date"])
    new_df = new_df[new_df["Date"]>pd.to_datetime(dates[0], format="%m-%Y")]
    new_df = new_df[new_df["Date"]<pd.to_datetime(dates[1], format="%m-%Y")]
    grouped_df = (new_df.drop("Date", axis = 1).sum()).T.reset_index(name="Amount").rename({"index": "Media"}, axis = 1)
    grouped_df["Media Type"] = [v.split("_")[1] for v in grouped_df["Media"]]
    grouped_df["Media Channel"] = [v.split("_")[2] for v in grouped_df["Media"]]
    grouped_df.drop("Media", axis = 1, inplace = True)
    return grouped_df[["Media Type", "Media Channel", "Amount"]]
    

def preprocess_budget(df: pd.DataFrame):
    new_df = df.reset_index().melt(id_vars = "index", value_vars = ["previous_budget", "optimized_budget"], var_name = "Budget", value_name = "Amount")
    new_df.insert(0, "Media Type", [w.split("_")[1] for w in new_df["index"]])
    new_df.rename({"index": "Media Channel"}, axis = 1, inplace = True)
    new_df["Media Channel"] = [c.split("_")[-1] for c in new_df["Media Channel"]]
    
    previous = new_df[new_df["Budget"] == "previous_budget"].reset_index(drop = True)
    previous["Allocation"] = previous["Amount"]/previous["Amount"].sum()
    
    optimized = new_df[new_df["Budget"] == "optimized_budget"].reset_index(drop = True)
    optimized["Allocation"] = optimized["Amount"]/optimized["Amount"].sum()
    return previous.drop(["Budget", "Amount"], axis = 1), optimized.drop(["Budget", "Amount"], axis = 1)

def preprocess_roi(df: pd.DataFrame):
    new_df = df.copy()
    new_df.insert(0, "Media Type", [w.split("_")[1] for w in new_df["index"]])
    new_df.rename({"index": "Media Channel"}, axis = 1, inplace = True)
    new_df["Media Channel"] = [c.split("_")[-1] for c in new_df["Media Channel"]]
    new_df.drop(["contribution", "confidence_int_inf_contribution", "confidence_int_sup_contribution"], axis = 1, inplace=True)
    return new_df
    

def get_budget(amount : float, 
               breakdown : pd.DataFrame):
    new_budget = breakdown.copy()
    allocations = new_budget["Amount"]/new_budget["Amount"].sum()
    new_budget["New Amount"] = allocations*amount
    new_budget["Allocation"] = allocations*100
    return new_budget

def access_type_budget(df, media_type, val_type):
    return df.loc[df["Media Type"] == media_type, [val_type]].values[0][0]

def access_channel_budget(df, media_channel, val_type):
    return df.loc[df["Media Channel"] == media_channel, [val_type]].values[0][0]

def get_channel_allocation(df : pd.DataFrame,
                           amount : float, 
                           media_type : str):
    new_budget = df.copy()
    new_budget = new_budget[new_budget["Media Type"] == media_type]
    allocations = new_budget["Amount"]/new_budget["Amount"].sum()
    new_budget["New Amount"] = allocations*amount
    new_budget["Allocation"] = allocations*100
    return new_budget

def saveTypeBudget(df : pd.DataFrame,
                   values : dict,
                   usePerc : bool,
                   budget : float):
    if usePerc == True:
        for k,v in values.items():
            df.loc[df["Media Channel"] ==k,["Allocation"]] = v
            df["New Amount"] = budget*df["Allocation"]/100
    else:
        for k,v in values.items():
            df.loc[df["Media Channel"]==k, ["New Amount"]] = v
            df["Allocation"] = df["New Amount"]/budget*100
    return df