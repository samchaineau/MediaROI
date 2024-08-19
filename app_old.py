import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

from tools import *

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
    
    st.header("Initial & Recos")
    initCol, recoCol = st.columns(2)
    
    with initCol:
        st.header("Initial budget & ROI")
            
    st.header("Simulations")
    optimCol, dateSelector, percCol = st.columns([3,5,3])
    
    with optimCol:
        useOptimized = st.toggle("Use optimized budget")
    
    with dateSelector:
        monthsList = get_months(raw_data["DATA"])
        startSelector = st.selectbox("Choose a start date", options=monthsList, index = len(monthsList)-12)
        endSelector = st.selectbox("Choose an end date", options=monthsList, index = len(monthsList)-1)
        raw_data = preprocess_data(raw_data["DATA"], [startSelector, endSelector])
    
    with percCol:
        usePerc = st.toggle("Use percentage for allocation", value=True)
    
    optimized_budget, previous_budget = preprocess_budget(mmm_results_data["budget_optimization"])
    optimized_budget["Amount"] = optimized_budget["Allocation"]*raw_data["Amount"].sum()
    previous_budget["Amount"] = previous_budget["Allocation"]*raw_data["Amount"].sum()
    
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
    
    if "initialBudget" not in st.session_state.keys():
        st.session_state.initialBudget = previous_budget    
    st.session_state.initialBudget["Amount"] = totalBudget * st.session_state.initialBudget["Allocation"]
    
    if "optimalBudget" not in st.session_state.keys():
        st.session_state.optimalBudget = optimized_budget    
    st.session_state.optimalBudget["Amount"] = totalBudget * st.session_state.optimalBudget["Allocation"]
    
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
            st.session_state.initialResults = 
            
            st.session_state.optimalResults = st.session_state.optimalBudget.merge(roi_values, how = "left", on = ["Media Type", "Media Channel"])[["Media Type", "Media Channel", "Amount", "Allocation", "roi"]]
            st.session_state.optimalResults.rename({"Amount" : "Investment", "Allocation" : "Allocation %"}, axis = 1, inplace = True)
            st.session_state.optimalResults["Contribution"] = st.session_state.optimalResults["roi"]*st.session_state.optimalResults["Investment"]
            
            st.session_state.finalSimulationData = pd.concat([df for df in st.session_state.simulation_data.values()]).reset_index(drop =True).drop(["Amount"], axis = 1).merge(roi_values, how = "left", on = ["Media Type", "Media Channel"])
            st.session_state.simulationResults = st.session_state.finalSimulationData[["Media Type", "Media Channel", "New Amount", "Allocation", "roi"]]
            st.session_state.simulationResults.rename({"New Amount" : "Investment", "Allocation" : "Allocation %"}, axis = 1, inplace = True)
            st.session_state.simulationResults["Contribution"] = st.session_state.simulationResults["roi"]*st.session_state.simulationResults["Investment"]
            
    if "simulationResults" in st.session_state.keys():
        csv = convert_df( st.session_state.simulationResults)
        with downloadCol:
            st.download_button(
                label="Download simulations",
                data=csv,
                file_name="simulation_df.csv",
                mime="text/csv")
        
        comparison = compare_global_roi({"Initial" : st.session_state.initialResults,
                                         "Optimal" : st.session_state.optimalResults,
                                         "Simulated" : st.session_state.simulationResults})
        figcomp = px.bar(comparison, x='Scenario', y='ROI', color="Scenario", text='Values')
        figcomp.update_traces(textposition='outside')
        st.plotly_chart(figcomp)
        
        simulatedExpander = st.expander("Simulated Scenario")
        with simulatedExpander:
            colBudget, colRoi = st.columns([2, 4])
            with colBudget:
                st.plotly_chart(px.pie(st.session_state.simulationResults.sort_values("Investment"), values='Investment', names='Media Type', color="Media Type"))
                
            with colRoi:
                grouped_contrib = st.session_state.simulationResults.drop(["roi", "Allocation %", "Media Channel"], axis = 1).groupby(["Media Type"]).sum().reset_index()
                grouped_contrib["ROI"] = grouped_contrib["Contribution"]/grouped_contrib["Investment"] 
                grouped_contrib_total = pd.DataFrame({'Media Type': ["Total"], "Investment": [sum(grouped_contrib["Investment"])], "Contribution": [sum(grouped_contrib["Contribution"])], "ROI": [sum(grouped_contrib["Contribution"])/sum(grouped_contrib["Investment"])]})
                grouped_contrib = pd.concat([grouped_contrib.sort_values("ROI"), grouped_contrib_total]).reset_index(drop=True)
                grouped_contrib['Values'] = grouped_contrib['ROI'].apply(lambda x: f'{x:.2f}')
                fig = px.bar(grouped_contrib, x='Media Type', y='ROI', color="Media Type", text='Values')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig)
        
        initialExpander = st.expander("Initial Scenario")
        with initialExpander:
            colBudget, colRoi = st.columns([2, 4])
            with colBudget:
                st.plotly_chart(px.pie(st.session_state.initialResults.sort_values("Investment"), values='Investment', names='Media Type', color="Media Type"))
                
            with colRoi:
                grouped_contrib = st.session_state.initialResults.drop(["roi", "Allocation %", "Media Channel"], axis = 1).groupby(["Media Type"]).sum().reset_index()
                grouped_contrib["ROI"] = grouped_contrib["Contribution"]/grouped_contrib["Investment"] 
                grouped_contrib_total = pd.DataFrame({'Media Type': ["Total"], "Investment": [sum(grouped_contrib["Investment"])], "Contribution": [sum(grouped_contrib["Contribution"])], "ROI": [sum(grouped_contrib["Contribution"])/sum(grouped_contrib["Investment"])]})
                grouped_contrib = pd.concat([grouped_contrib.sort_values("ROI"), grouped_contrib_total]).reset_index(drop=True)
                grouped_contrib['Values'] = grouped_contrib['ROI'].apply(lambda x: f'{x:.2f}')
                fig = px.bar(grouped_contrib, x='Media Type', y='ROI', color="Media Type", text='Values')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig)
        
        optimalExpander = st.expander("Optimal Scenario")
        with optimalExpander:
            colBudget, colRoi = st.columns([2, 4])
            with colBudget:
                st.plotly_chart(px.pie(st.session_state.optimalResults.sort_values("Investment"), values='Investment', names='Media Type', color="Media Type"))
                
            with colRoi:
                grouped_contrib = st.session_state.optimalResults.drop(["roi", "Allocation %", "Media Channel"], axis = 1).groupby(["Media Type"]).sum().reset_index()
                grouped_contrib["ROI"] = grouped_contrib["Contribution"]/grouped_contrib["Investment"] 
                grouped_contrib_total = pd.DataFrame({'Media Type': ["Total"], "Investment": [sum(grouped_contrib["Investment"])], "Contribution": [sum(grouped_contrib["Contribution"])], "ROI": [sum(grouped_contrib["Contribution"])/sum(grouped_contrib["Investment"])]})
                grouped_contrib = pd.concat([grouped_contrib.sort_values("ROI"), grouped_contrib_total]).reset_index(drop=True)
                grouped_contrib['Values'] = grouped_contrib['ROI'].apply(lambda x: f'{x:.2f}')
                fig = px.bar(grouped_contrib, x='Media Type', y='ROI', color="Media Type", text='Values')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig)
        
if __name__ == "__main__":
    main()