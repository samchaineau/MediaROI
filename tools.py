import numpy as np
import pandas as pd
from lightweight_mmm import media_transforms

def load_excel(file):
    excel_data = pd.read_excel(file, sheet_name=None)
    return excel_data

def filter_data(df : pd.DataFrame,
                    dates : list):
    new_df = df.copy()
    new_df["Date"] = pd.to_datetime(new_df["Date"], dayfirst=True)
    new_df = new_df[new_df["Date"]>dates[0]]
    new_df = new_df[new_df["Date"]<dates[1]]
    return new_df
    
def preprocess_data(df : pd.DataFrame,
                    dates : list):
    new_df = df.copy()
    new_df.drop(["index"], axis = 1, inplace=True)
    to_keep = [c for c in new_df.columns if "Media" in c]+["Date"]
    new_df = new_df[to_keep]
    new_df = filter_data(new_df, dates)
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

def compare_global_roi(dict_of_df : dict):
    roi_per_df = {k : [sum(v["Contribution"])/sum(v["Investment"])] for k,v in dict_of_df.items()}
    compare_df = pd.DataFrame(roi_per_df).T.reset_index()
    compare_df.columns = ["Scenario", "ROI"]
    compare_df['Values'] = compare_df['ROI'].apply(lambda x: f'{x:.2f}')
    return compare_df


def makeSimulation(raw_data_period : pd.DataFrame,
                   budget_allocation : pd.DataFrame,
                   target_scaler,
                   media_scaler,
                   model):
    
            simulationData = budget_allocation.copy()
            simulationData["Media_var"] = ["Media_" +c["Media Type"]+"_"+c["Media Channel"]+"_€" if c["Media Channel"] not in ["Search", "Social", "Display"] else "Media_" +c["Media Type"]+"_€" for c in budget_allocation.to_dict(orient = "records")]
            simulationRecords = simulationData[["Media_var", "Amount"]].set_index('Media_var')['Amount'].to_dict()
            
            data_to_test = raw_data_period[[c for c in raw_data_period.columns if "Media" in c]]
            data_to_test = data_to_test/data_to_test.sum(axis = 0)         
            for c in data_to_test.columns:
                data_to_test[c] = data_to_test[c]*simulationRecords[c]
            
            data_to_test = data_to_test.to_numpy()
            carryover_list = [media_transforms.carryover(media_scaler.transform(data_to_test), model.trace["ad_effect_retention_rate"][i], model.trace["peak_effect_delay"][i]) for i in range(model.trace["ad_effect_retention_rate"].shape[0])]
            transformed_data_list = [media_transforms.apply_exponent_safe(data = carryover_list[i], exponent = model.trace["exponent"][i]) for i in range(model.trace["exponent"].shape[0])]
            contribution_list = [model.trace["coef_media"][i] * transformed_data_list[i] for i in range(len(model.trace["coef_media"]))]
            contributionSimulated = target_scaler.inverse_transform(np.array(contribution_list).mean(axis = 0))
            contributionSimulated = pd.DataFrame(contributionSimulated, columns = [c for c in raw_data_period.columns if "Media" in c]).fillna(0).sum().reset_index(name = "Contribution").rename({"index" : "Media_var"}, axis = 1)
            contributionSimulated = simulationData.merge(contributionSimulated, on = "Media_var", how = "left")
            return contributionSimulated