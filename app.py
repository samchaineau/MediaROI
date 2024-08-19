import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from lightweight_mmm import utils, optimize_media, preprocessing, media_transforms, optimize_media
import jax.numpy as jnp
from fpdf import FPDF
import kaleido

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
    excel_data = load_excel("Raw_Data_mediaROI_simulateur - mai 2024.xlsx")
    train_split = round(0.9*excel_data["DATA"].shape[0])
    media_data_train = excel_data["DATA"][[c for c in excel_data["DATA"].columns if "Media_" in c]]
    media_data_train = media_data_train.to_numpy()[:train_split, ...]
    target_train = excel_data["DATA"]['Ventes'].to_numpy(dtype='float32')[:train_split]
    
    
    
    if "model" not in st.session_state.keys():
        st.session_state.model = utils.load_model(file_path= "media_mix_model.pkl")
        st.session_state.media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
        st.session_state.media_scaler.fit_transform(media_data_train)
        st.session_state.target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
        st.session_state.target_scaler.fit_transform(target_train)
    
    mmm_results_data = load_excel("Resultats MMM pour simulateur - mediaROI - mai 2024.xlsx")

    
    st.image("logo mediaROI.png")
    
    with st.container():
        startDateCol, endDateCol, _, percCol, _ = st.columns([2,2,2,3,4])
        
        with percCol:
            usePerc = st.toggle("Use percentage for allocation", value=True)
    
    previous_budget, optimized_budget = preprocess_budget(mmm_results_data["budget_optimization"])
    
    optimized_budget = makeSimulation(optimized_budget,
                                      st.session_state.target_scaler,
                                      st.session_state.media_scaler,
                                      st.session_state.model)
    optimized_budget["Budget"] = "Optimal"
    grouped_optimized = optimized_budget[["Media Type", "Amount", "Allocation", "Contribution"]].groupby("Media Type").sum().reset_index()
    
    previous_budget = makeSimulation(previous_budget,
                                     st.session_state.target_scaler,
                                     st.session_state.media_scaler,
                                     st.session_state.model)
    previous_budget["Budget"] = "Initial"
    grouped_previous = previous_budget[["Media Type", "Amount", "Allocation", "Contribution"]].groupby("Media Type").sum().reset_index()
    
    initialROI = (sum(grouped_previous["Contribution"])/sum(grouped_previous["Amount"]))
    optimizedROI = (sum(grouped_optimized["Contribution"])/sum(grouped_optimized["Amount"]))
    evolInitOptim = (optimizedROI/initialROI)-1
    incremVal = sum(grouped_optimized["Contribution"])-sum(grouped_previous["Contribution"])
    
    st.header("Mix Media Initial")
    
    initAlloc, initROI, _, _ = st.columns([9,2,2,2])
    initMedia =  grouped_previous.to_dict(orient = "records")
    initMedia  = {v["Media Type"] : v for v in initMedia}
    toDisplayMedia = ["TV", "OOH", "Radio", "Print", "Search", "Display", "VOL", "Social"]
    
    with st.container():
        with initAlloc:
            initMediaCol = st.columns(8)
            initMediaCol = list(zip(toDisplayMedia, initMediaCol))
            for mediaName, pmediaCol in initMediaCol:
                with pmediaCol:
                    if mediaName in initMedia.keys():
                        if usePerc:
                            st.number_input(f'Initial {mediaName} in %', value=initMedia[mediaName]["Allocation"]*100, disabled=True)
                        else:
                            st.number_input(f'Initial {mediaName} in €', value=initMedia[mediaName]["Amount"], disabled=True)
                    else:
                        if usePerc:
                            st.number_input(f'Initial {mediaName} in %', value=0, disabled=True)
                        else:
                            st.number_input(f'Initial {mediaName} in €', value=0, disabled=True)

        with initROI:
            st.markdown(
                        """
                        <div style="text-align: center;">
                            <h3>ROI</h3>
                            <h3>{:.2f}</h3>
                        </div>
                        """.format(initialROI),
                        unsafe_allow_html=True
                        )
            
    st.header("Mix Media Optimal")
    optimAlloc, optimROI, evolROICol, incremColVal = st.columns([9,2,2,2])
    optimMedia =  grouped_optimized.to_dict(orient = "records")
    optimMedia  = {v["Media Type"] : v for v in optimMedia}
    
    with st.container():
        with optimAlloc:
            optimMediaCol = st.columns(8)
            optimMediaCol = list(zip(toDisplayMedia, optimMediaCol))
            for mediaName, omediaCol in optimMediaCol:
                with omediaCol:
                    if mediaName in optimMedia.keys():
                        if usePerc:
                            st.number_input(f'Optimal {mediaName} in %', value=optimMedia[mediaName]["Allocation"]*100, disabled=True)
                        else:
                            st.number_input(f'Optimal {mediaName} in €', value=optimMedia[mediaName]["Amount"], disabled=True)
                    else:
                        if usePerc:
                            st.number_input(f'Optimal {mediaName} in %', value=0, disabled=True)
                        else:
                            st.number_input(f'Optimal {mediaName} in €', value=0, disabled=True)
        with optimROI:
            st.markdown(
                        """
                        <div style="text-align: center;">
                            <h3>ROI</h3>
                            <h3>{:.2f}</h3>
                        </div>
                        """.format(optimizedROI),
                        unsafe_allow_html=True
                        )
        
        with evolROICol:
            st.markdown(
                        """
                        <div style="text-align: center;">
                            <h3>ROI evolution</h3>
                            <h3>{:.2f} %</h3>
                        </div>
                        """.format(evolInitOptim),
                        unsafe_allow_html=True
                        )
        
        with incremColVal:
            st.markdown(
                        """
                        <div style="text-align: center;">
                            <h3>Incremental</h3>
                            <h3>{:.0f} (€)</h3>
                        </div>
                        """.format(incremVal),
                        unsafe_allow_html=True
                        )
    
    
    st.title("Simulateur Mix Media")
    optimCol, _, _ = st.columns([3,5,3])
    
    with optimCol:
        useOptimized = st.toggle("Use optimized budget")
    
    with st.container():
        simAlloc, simROI, evolSimROI, simColVal = st.columns([9,2,2,2])
        with simAlloc:
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
                grouped_start_amount = start_amount.drop(["Media Channel", "Budget", "Amount"], axis = 1).groupby("Media Type").sum().reset_index()
            
            media_cols_pairs = list(zip(toDisplayMedia, st.columns(8)))
            mediaCol = {e[0] : {"col" : e[1]} for e in media_cols_pairs}
            
            for media, elements in mediaCol.items():
                with elements["col"]:
                    if media in list(grouped_start_amount["Media Type"]):
                        if usePerc:
                            elements["input"] = st.number_input(f"{media} in %", value=access_type_budget(grouped_start_amount, media, "Allocation"))
                            elements["data"] = get_channel_allocation(start_amount, elements["input"]*totalBudget/100, media)
                        else:
                            elements["input"] = st.number_input(f"{media} in Euros", value=access_type_budget(grouped_start_amount, media, "New Amount"))
                            elements["data"] = get_channel_allocation(start_amount, elements["input"], media).reset_index(drop=True)
                    else:
                        if usePerc:
                            elements["input"] = st.number_input(f"{media} in %", value=0, disabled = True)
                        else:
                            elements["input"] = st.number_input(f"{media} in Euros", value=0, disabled = True)
    
    saveGlobal, _ = st.columns([4, 1])
    with saveGlobal:
        saveButtonGlobal = st.button("Save budget")
        
        if saveButtonGlobal:
            st.session_state.simulation_data = {k: v["data"] for k,v in mediaCol.items() if "data" in v.keys()}
            st.session_state.totalBudget = totalBudget
    
    if usePerc==False:
        currentBudget = round(sum([e["input"] for e in mediaCol.values()]))
        if currentBudget != round(totalBudget):
            st.warning(f"Total budget is not equal to media allocations (current total : {round(currentBudget, 2)}). Please check values or switch to percentages")
    else:
        currentAll = round(sum([e["input"] for e in mediaCol.values()]))
        if currentAll != 100:
            st.warning(f"Total budget allocation is different from 100% (current total : {round(currentAll, 2)}%). Please check allocation.")
    
    if "simulation_data" not in st.session_state.keys():    
        st.session_state.simulation_data = {k: v["data"] for k,v in mediaCol.items() if "data" in v.keys()}
    
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
    
    _, downloadExcelCol, downloadPDFCol, simulateCol = st.columns([4, 1, 1, 1])
    
    with simulateCol:
        simulateButton = st.button("Simulate results")
        if simulateButton:            
            st.session_state.finalSimulationData = pd.concat([df for df in st.session_state.simulation_data.values()]).reset_index(drop =True).drop(["Amount"], axis = 1)[["Media Type", "Media Channel", "New Amount"]]
            st.session_state.finalSimulationData = st.session_state.finalSimulationData.rename({"New Amount" : "Amount"}, axis = 1)
            st.session_state.finalSimulationData = optimized_budget[["Media Type", "Media Channel"]].merge(st.session_state.finalSimulationData, on = ["Media Type", "Media Channel"], how = "left")
            
            st.session_state.simulationResults = makeSimulation(st.session_state.finalSimulationData, 
                                                                st.session_state.target_scaler,
                                                                st.session_state.media_scaler,
                                                                st.session_state.model)
            
            st.session_state.simulationResults["Budget"] = "Simulated"
            st.session_state.simulationResults["Allocation"] = st.session_state.simulationResults["Amount"]/st.session_state.simulationResults["Amount"].sum()
    
    if "simulationResults" in st.session_state.keys():
        csv = convert_df(st.session_state.simulationResults)
        with downloadExcelCol:
            st.download_button(
                label="Download simulations",
                data=csv,
                file_name="simulation_df.csv",
                mime="text/csv")
        
        roiTab, contTab, budTab = st.tabs(["ROI", "Contribution", "Budget"])
        
        init_optim_df = pd.concat([optimized_budget, previous_budget, st.session_state.simulationResults]).reset_index(drop=True)
        
        figs = []
        with budTab:
            comparison_df = init_optim_df[["Media Type", "Budget", "Amount"]]
            comparison_df = comparison_df.groupby(["Media Type", "Budget"]).sum().reset_index()
            comparison_df["Amount"] = comparison_df['Amount']/1000
            comparison_df['Values'] = comparison_df['Amount'].apply(lambda x: f'{x:,.2f}')
            fig = px.bar(comparison_df, 
                            x='Media Type', 
                            y='Amount', 
                            color="Budget", 
                            text='Values', 
                            barmode='group', 
                            labels = {"Amount" : "Investment (k Euros)"},
                            color_discrete_sequence=["#000000", "#070996", "#19ACBF"])
            fig.update_traces(textposition='outside')
            figs.append(fig)
            st.plotly_chart(fig)
        
        with contTab:
            comparison_df = init_optim_df[["Media Type", "Budget", "Contribution"]]
            comparison_df = comparison_df.groupby(["Media Type", "Budget"]).sum().reset_index()
            comparison_df["Contribution"] = comparison_df['Contribution']/1000
            comparison_df['Values'] = comparison_df['Contribution'].apply(lambda x: f'{x:,.2f}')
            fig = px.bar(comparison_df, 
                            x='Media Type', 
                            y='Contribution', 
                            color="Budget", 
                            text='Values', 
                            barmode='group', 
                            labels = {"Contribution" : "Contribution (k Euros)"}, 
                            color_discrete_sequence=["#000000", "#070996", "#19ACBF"])
            fig.update_traces(textposition='outside')
            figs.append(fig)
            st.plotly_chart(fig)
                
        with roiTab:
            comparison_df = init_optim_df[["Media Type", "Budget", "Amount", "Contribution"]]
            comparison_df = comparison_df.groupby(["Media Type", "Budget"]).sum().reset_index()
            comparison_df["ROI"] = comparison_df["Contribution"]/comparison_df["Amount"]
            comparison_df['Values'] = comparison_df['ROI'].apply(lambda x: f'{x:,.2f}')
            fig = px.bar(comparison_df, 
                            x='Media Type', 
                            y='ROI', 
                            color="Budget", 
                            text='Values', 
                            barmode='group',
                            color_discrete_sequence=["#000000", "#070996", "#19ACBF"])
            fig.update_traces(textposition='outside')
            figs.append(fig)
            st.plotly_chart(fig)
        
        image_files = ['plot1.png', 'plot2.png', 'plot3.png']
        
        for fig, filename in zip(figs, image_files):
            fig.write_image(filename)
            
        def make_pdf(image_files, output_filename):
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.set_auto_page_break(0)
            for image_file in image_files:
                pdf.add_page()
                pdf.image(image_file, x=10, y=10, w=277)
            pdf.output(output_filename)


        pdf_filename = 'simulateur_rapport.pdf'
        make_pdf(image_files, pdf_filename)
        
        with open("simulateur_rapport.pdf", "rb") as pdf_file:
            document = pdf_file.read()
        
        with downloadPDFCol:
            st.download_button(
                label="Download PDF",
                data=document,
                file_name='simulateur_rapport.pdf',
                mime='application/pdf'
            )
        
        roi_simulation = (sum(st.session_state.simulationResults["Contribution"])/sum(st.session_state.simulationResults["Amount"]))
        roi_evol_simulation_perc = (roi_simulation/initialROI)-1
        increm_simulation_val = sum(st.session_state.simulationResults["Contribution"]) - sum(grouped_previous["Contribution"])
        
        with simROI:
            st.markdown(
                        """
                        <div style="text-align: center;">
                            <h3>ROI</h3>
                            <h3>{:.2f}</h3>
                        </div>
                        """.format(roi_simulation),
                        unsafe_allow_html=True
                        )
            
        with evolSimROI:
            st.markdown(
                        """
                        <div style="text-align: center;">
                            <h3>ROI evolution</h3>
                            <h3>{:.2f} %</h3>
                        </div>
                        """.format(roi_evol_simulation_perc),
                        unsafe_allow_html=True
                        )
        
        with simColVal:
            st.markdown(
                        """
                        <div style="text-align: center;">
                            <h3>Incremental</h3>
                            <h3>{:.0f} (€)</h3>
                        </div>
                        """.format(increm_simulation_val),
                        unsafe_allow_html=True
                        )
    
        
         
        
if __name__ == "__main__":
    main()