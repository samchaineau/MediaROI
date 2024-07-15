import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

from tools import *

st.set_page_config(layout="wide")

def reset_channel_budget():
    st.session_state.channel_budget = {}
    
@st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8")

def get_months(df : pd.DataFrame):
    new_df = df.copy()
    new_df['Date'] = pd.to_datetime(new_df['Date'], format='%d/%m/%Y')
    unique_months = new_df['Date'].dt.to_period('M').unique()
    unique_months = pd.to_datetime(unique_months.to_timestamp()).date
    return list(unique_months)

def main():
    st.title("MediaROI simulator")
    raw_data = load_excel("Raw_Data_mediaROI_simulateur - mai 2024.xlsx")
    mmm_results_data = load_excel("Resultats MMM pour simulateur - mediaROI - mai 2024.xlsx")
    roi_values = preprocess_roi(mmm_results_data["mmm_results"])
    
    monthsList = get_months(raw_data["DATA"])
    raw_data = preprocess_data(raw_data["DATA"], [monthsList[-12], monthsList[-1]])
    
    optimized_budget, previous_budget = preprocess_budget(mmm_results_data["budget_optimization"])
    optimized_budget["Amount"] = optimized_budget["Allocation"]*raw_data["Amount"].sum()
    optimized_budget = optimized_budget.merge(roi_values, on = ["Media Type", "Media Channel"])
    optimized_budget["Contribution"] = optimized_budget["Amount"]*optimized_budget["roi"]
    optimized_budget["Budget"] = "Optimal"
    
    previous_budget["Amount"] = previous_budget["Allocation"]*raw_data["Amount"].sum()
    previous_budget = previous_budget.merge(roi_values, on = ["Media Type", "Media Channel"])
    previous_budget["Contribution"] = previous_budget["Amount"]*previous_budget["roi"]
    previous_budget["Budget"] = "Initial"
    
    _, initCol, _, recoCol, _ = st.columns([1,2,1,2,1])
    with initCol:
        st.header("Initial scenario")
        st.markdown(f"Initial ROI is")
        st.subheader(f'{(sum(previous_budget["Contribution"])/sum(previous_budget["Amount"])):,.2f}')
        
    with recoCol:
        st.header("Optimal scenario")
        st.markdown(f"Optimal ROI is")
        st.subheader(f'{(sum(optimized_budget["Contribution"])/sum(optimized_budget["Amount"])):,.2f}')
    
    _, centerCol, _ = st.columns([4,3,4])
    with centerCol:   
        st.markdown("Incremental")
        st.subheader(f'{(sum(optimized_budget["Contribution"]) - sum(previous_budget["Contribution"])):,.2f} Euros')
        
    st.header("Simulations")
    optimCol, _, percCol = st.columns([3,5,3])
    
    with optimCol:
        useOptimized = st.toggle("Use optimized budget")
    
    with percCol:
        usePerc = st.toggle("Use percentage for allocation", value=True)
    
    if useOptimized:
        if "totalBudget" not in st.session_state.keys():
            totalBudget = st.number_input("Total budget to simulate", value = optimized_budget["Amount"].sum())
        else:
            totalBudget = st.number_input("Total budget to simulate", value = st.session_state.totalBudget)
        start_amount = get_budget(totalBudget, optimized_budget)
        grouped_start_amount = start_amount.drop(["Media Channel", "Amount"], axis = 1).groupby("Media Type").sum().reset_index()
    else:
        if "totalBudget" not in st.session_state.keys():
            totalBudget = st.number_input("Total budget to simulate", value = previous_budget["Amount"].sum())
        else:
            totalBudget = st.number_input("Total budget to simulate", value = st.session_state.totalBudget)
        start_amount = get_budget(totalBudget, previous_budget)
        grouped_start_amount = start_amount.drop(["Media Channel", "Amount"], axis = 1).groupby("Media Type").sum().reset_index()
    
    media_cols_pairs = list(zip(list(grouped_start_amount["Media Type"]), st.columns(grouped_start_amount.shape[0])))
    mediaCol = {e[0] : {"col" : e[1]} for e in media_cols_pairs}
    
    for media, elements in mediaCol.items():
        with elements["col"]:
            if usePerc:
                elements["input"] = st.number_input(f"{media} in %", value=access_type_budget(grouped_start_amount, media, "Allocation"))
                elements["data"] = get_channel_allocation(start_amount, elements["input"]*totalBudget/100, media)
            else:
                elements["input"] = st.number_input(f"{media} in Euros", value=access_type_budget(grouped_start_amount, media, "New Amount"))
                elements["data"] = get_channel_allocation(start_amount, elements["input"], media).reset_index(drop=True)
    
    _, saveGlobal = st.columns([4, 1])
    with saveGlobal:
        saveButtonGlobal = st.button("Save budget")
        
        if saveButtonGlobal:
            st.session_state.simulation_data = {k: v["data"] for k,v in mediaCol.items()}
            st.session_state.totalBudget = totalBudget
    
    if usePerc==False:
        currentBudget = round(sum([e["input"] for e in mediaCol.values()]))
        if currentBudget != totalBudget:
            st.warning(f"Total budget is not equal to media allocations (current total : {round(currentBudget, 2)}). Please check values or switch to percentages")
    else:
        currentAll = round(sum([e["input"] for e in mediaCol.values()]))
        if currentAll != 100:
            st.warning(f"Total budget allocation is different from 100% (current total : {round(currentAll, 2)}%). Please check allocation.")
    
    if "simulation_data" not in st.session_state.keys():    
        st.session_state.simulation_data = {k: v["data"] for k,v in mediaCol.items()}
    
    budgetExpander = st.expander("Advanced settings")
    
    with budgetExpander:
        colSelect, typeBudget, percAdvanced = st.columns([3,3,3])
        
        with colSelect:
            typeSelector = st.selectbox(label="Media", options=list(st.session_state.simulation_data.keys()), on_change=reset_channel_budget())
            
        with typeBudget:
            budget_sum = st.session_state.simulation_data[typeSelector]["New Amount"].sum()
            st.markdown(f"The budget allocated to {typeSelector} is {round(budget_sum, 2)}")
        
        with percAdvanced:
            useAdvPerc = st.toggle("Use percentage for advanced allocation", value = usePerc)
        
        for channel in  st.session_state.simulation_data[typeSelector]["Media Channel"]:
            if useAdvPerc:
                newInput = st.number_input(f"Allocation in % for {channel}", value = access_channel_budget(st.session_state.simulation_data[typeSelector], channel, "Allocation"))
            else:
                newInput = st.number_input(f"Allocation in euros for {channel}", value = access_channel_budget(st.session_state.simulation_data[typeSelector], channel, "New Amount"))
            st.session_state.channel_budget.update({channel : newInput})
            
        if useAdvPerc == False:
            if round(budget_sum) != round(sum(st.session_state.channel_budget.values())):
                st.warning(f"Total budget is not equal to media allocations ({round(sum(st.session_state.channel_budget.values()), 2)}). Please check values or switch to percentages")
        else:
            if round(sum(st.session_state.channel_budget.values())) != 100:
                st.warning(f"Total budget allocation is greater than 100% ({round(sum(st.session_state.channel_budget.values()), 2)}). Please check allocation.")
        
        _, saveChannel = st.columns([4, 1])
        with saveChannel:
            saveButtonChannel = st.button("Save channel budget")
            
            if saveButtonChannel:
                st.session_state.simulation_data[typeSelector] = saveTypeBudget(st.session_state.simulation_data[typeSelector], st.session_state.channel_budget, usePerc=useAdvPerc, budget=budget_sum)
    
    _, downloadCol, simulateCol = st.columns([4, 1, 1])
    
    with simulateCol:
        simulateButton = st.button("Simulate results")
        if simulateButton:            
            st.session_state.finalSimulationData = pd.concat([df for df in st.session_state.simulation_data.values()]).reset_index(drop =True).drop(["Amount"], axis = 1)
            st.session_state.simulationResults = st.session_state.finalSimulationData[["Media Type", "Media Channel", "Allocation", "New Amount", "roi", "confidence_int_inf_roi", "confidence_int_sup_roi"]]
            st.session_state.simulationResults.rename({"New Amount" : "Amount"}, axis = 1, inplace = True)
            st.session_state.simulationResults["Allocation"] = st.session_state.simulationResults["Amount"]/st.session_state.simulationResults["Amount"].sum()
            st.session_state.simulationResults["Contribution"] = st.session_state.simulationResults["roi"]*st.session_state.simulationResults["Amount"]
            st.session_state.simulationResults["Budget"] = "Simulated"
            
    if "simulationResults" in st.session_state.keys():
        csv = convert_df( st.session_state.simulationResults)
        with downloadCol:
            st.download_button(
                label="Download simulations",
                data=csv,
                file_name="simulation_df.csv",
                mime="text/csv")
        
        filterCol, simCol = st.columns([1, 3])
        
        with filterCol:
            budg_or_cont = st.selectbox(label = "Select indicator to compare previous and optimal scenario",
                                        options = ["Budget", "Contribution", "ROI"], 
                                        index = 2)
        
        
        init_optim_df = pd.concat([optimized_budget, previous_budget, st.session_state.simulationResults]).reset_index(drop=True)
        
        with simCol:
            if budg_or_cont == "Budget":
                comparison_df = init_optim_df[["Media Type", "Budget", "Amount"]]
                comparison_df = comparison_df.groupby(["Media Type", "Budget"]).sum().reset_index()
                comparison_df["Amount"] = comparison_df['Amount']/1000
                comparison_df['Values'] = comparison_df['Amount'].apply(lambda x: f'{x:,.2f}')
                fig = px.bar(comparison_df, x='Media Type', y='Amount', color="Budget", text='Values', barmode='group', labels = {"Amount" : "Investment (k Euros)"})
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig)
                
            
            if budg_or_cont == "Contribution":
                comparison_df = init_optim_df[["Media Type", "Budget", "Contribution"]]
                comparison_df = comparison_df.groupby(["Media Type", "Budget"]).sum().reset_index()
                comparison_df["Contribution"] = comparison_df['Contribution']/1000
                comparison_df['Values'] = comparison_df['Contribution'].apply(lambda x: f'{x:,.2f}')
                fig = px.bar(comparison_df, x='Media Type', y='Contribution', color="Budget", text='Values', barmode='group', labels = {"Contribution" : "Contribution (k Euros)"})
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig)
                
            if budg_or_cont == "ROI":
                comparison_df = init_optim_df[["Media Type", "Budget", "Amount", "Contribution"]]
                comparison_df = comparison_df.groupby(["Media Type", "Budget"]).sum().reset_index()
                comparison_df["ROI"] = comparison_df["Contribution"]/comparison_df["Amount"]
                comparison_df['Values'] = comparison_df['ROI'].apply(lambda x: f'{x:,.2f}')
                fig = px.bar(comparison_df, x='Media Type', y='ROI', color="Budget", text='Values', barmode='group')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig)
    
        
         
        
if __name__ == "__main__":
    main()