import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from lightweight_mmm import (
    utils,
    preprocessing,
)
import jax.numpy as jnp
from fpdf import FPDF

from tools import *

st.set_page_config(layout="wide")


def reset_channel_budget():
    """Resets the session state for channel budget."""
    st.session_state.channel_budget = {}


@st.cache_data
def convert_df(df):
    """Converts a DataFrame into a CSV format."""
    return df.to_csv().encode("utf-8")


def main():
    # Retrieve parameters from the URL query
    session_params = st.query_params

    # Load custom CSS
    load_css("mediaroi-charte.css")

    # Load Excel data
    excel_data = load_excel("projets/" + session_params["mission"] + "/raw.xlsx")

    # Ajouter le titre de la page avec le nom de la mission
    st.title(f"Projet MMM : {session_params['mission']}")

    # Split the dataset based on the same split made during training
    train_split = round(0.9 * excel_data["DATA"].shape[0])

    # Extract and process media data
    media_data_train = excel_data["DATA"][
        [c for c in excel_data["DATA"].columns if "Media_" in c]
    ]
    media_data_train = media_data_train.to_numpy()[:train_split, ...]

    # Target variable (e.g., "Ventes")
    target_train = excel_data["DATA"]["Ventes"].to_numpy(dtype="float32")[:train_split]

    # Load model and scalers into session state
    if "model" not in st.session_state.keys():
        st.session_state.model = utils.load_model(
            file_path="projets/" + session_params["mission"] + "/model.pkl"
        )
        st.session_state.media_scaler = preprocessing.CustomScaler(
            divide_operation=jnp.mean
        )
        st.session_state.media_scaler.fit_transform(media_data_train)
        st.session_state.target_scaler = preprocessing.CustomScaler(
            divide_operation=jnp.mean
        )
        st.session_state.target_scaler.fit_transform(target_train)

    # Load MMM results data (this excel is the output of the modelling notebook)
    mmm_results_data = load_excel(
        "projets/" + session_params["mission"] + "/results.xlsx"
    )

    # Process previous and optimized budgets from the results
    previous_budget, optimized_budget = preprocess_budget(
        mmm_results_data["budget_optimization"]
    )

    # Get the contribution with the model from the optimized budget
    optimized_budget = makeSimulation(
        optimized_budget,
        st.session_state.target_scaler,
        st.session_state.media_scaler,
        st.session_state.model,
    )
    optimized_budget["Budget"] = "Optimal"

    # Group and summarize the optimized budget by media type
    grouped_optimized = (
        optimized_budget[["Media Type", "Amount", "Allocation", "Contribution"]]
        .groupby("Media Type")
        .sum()
        .reset_index()
    )

    # Simulate on the previous budget with the model
    previous_budget = makeSimulation(
        previous_budget,
        st.session_state.target_scaler,
        st.session_state.media_scaler,
        st.session_state.model,
    )
    previous_budget["Budget"] = "Initial"

    # Group and summarize the previous budget by media type
    grouped_previous = (
        previous_budget[["Media Type", "Amount", "Allocation", "Contribution"]]
        .groupby("Media Type")
        .sum()
        .reset_index()
    )

    # Calculate initial ROI and optimized ROI
    initialROI = sum(grouped_previous["Contribution"]) / sum(grouped_previous["Amount"])
    optimizedROI = sum(grouped_optimized["Contribution"]) / sum(
        grouped_optimized["Amount"]
    )
    evolInitOptim = (optimizedROI / initialROI) - 1  # Relative improvement in ROI

    # Incremental contribution value
    incremVal = sum(grouped_optimized["Contribution"]) - sum(
        grouped_previous["Contribution"]
    )

    with st.container():
        firstheader, _, percCol, _ = st.columns([8, 2, 8, 6])

        with firstheader:
            st.header("Mix Media Initial")

        with percCol:
            # Toggle for allocation input in percentage or absolute amount
            st.write("")
            usePerc = st.toggle("Use % for allocation", value=True)

    # Display the initial media allocation and ROI
    initAlloc, initROI, _, _ = st.columns([10, 2, 3, 3])
    initMedia = grouped_previous.to_dict(orient="records")
    initMedia = {v["Media Type"]: v for v in initMedia}

    # Media types to display
    toDisplayMedia = [
        "TV",
        "OOH",
        "RADIO",
        "PRINT",
        "SEARCH",
        "DISPLAY",
        "VOL",
        "SOCIAL",
    ]

    if len([m for m in initMedia.keys() if m not in toDisplayMedia]) > 1:
        st.warning(
            "Variables in files don't match taxonomy. Please make sure format are respected. Error variables are :"
        )
        st.write([m for m in initMedia.keys() if m not in toDisplayMedia[:4]])
        err1 = 1
    else:
        err1 = 0

    # Display allocation inputs for first half of media channels
    with initAlloc:
        with st.container():
            initMediaCol1 = st.columns(4)
            initMediaCol1 = list(zip(toDisplayMedia[:4], initMediaCol1))
            for mediaName, pmediaCol in initMediaCol1:
                with pmediaCol:
                    if mediaName in initMedia.keys():
                        # Display allocation in percentage or absolute amount
                        if usePerc:
                            st.number_input(
                                f"Init. {mediaName} in %",
                                value=initMedia[mediaName]["Allocation"] * 100,
                                disabled=True,
                            )
                        else:
                            st.number_input(
                                f"Init. {mediaName} in €",
                                value=initMedia[mediaName]["Amount"],
                                disabled=True,
                            )
                    else:
                        # Handle media channels not present in the data
                        if usePerc:
                            st.number_input(
                                f"Init. {mediaName} in %", value=0, disabled=True
                            )
                        else:
                            st.number_input(
                                f"Init. {mediaName} in €", value=0, disabled=True
                            )

        # Repeat the process for second half of media channels
        with st.container():
            initMediaCol2 = st.columns(4)
            initMediaCol2 = list(zip(toDisplayMedia[4:], initMediaCol2))
            for mediaName, pmediaCol in initMediaCol2:
                with pmediaCol:
                    if mediaName in initMedia.keys():
                        if usePerc:
                            st.number_input(
                                f"Init. {mediaName} in %",
                                value=initMedia[mediaName]["Allocation"] * 100,
                                disabled=True,
                            )
                        else:
                            st.number_input(
                                f"Init. {mediaName} in €",
                                value=initMedia[mediaName]["Amount"],
                                disabled=True,
                            )
                    else:
                        if usePerc:
                            st.number_input(
                                f"Init. {mediaName} in %", value=0, disabled=True
                            )
                        else:
                            st.number_input(
                                f"Init. {mediaName} in €", value=0, disabled=True
                            )

    # Display the calculated ROI in a styled container
    with initROI:
        with st.container():
            st.empty()
        with st.container():
            st.write("")
            st.markdown(
                f"""
                    <div style="
                        width: 100%;
                        text-align: center; 
                        border: 2px solid black; 
                        padding: 0px;
                        border-radius: 10px; 
                        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                    ">
                        <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">ROI</span>
                        <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">{initialROI:.2f}</span>
                    </div>
                    """,
                unsafe_allow_html=True,
            )

    # Set header for the section "Mix Media Optimal"
    st.header("Mix Media Optimal (constant budget)")

    # Create columns for optimal allocation, ROI, evolution of ROI, and incremental value
    optimAlloc, optimROI, evolROICol, incremColVal = st.columns([10, 2, 3, 3])

    # Convert the optimized media data into a dictionary format for easy lookup
    optimMedia = grouped_optimized.to_dict(orient="records")
    optimMedia = {v["Media Type"]: v for v in optimMedia}

    # First container for media allocation display (4 media types)
    with optimAlloc:
        with st.container():
            # Create columns for the first 4 media types
            optimMediaCol1 = st.columns(4)
            optimMediaCol1 = list(zip(toDisplayMedia[:4], optimMediaCol1))

            # Iterate through the first 4 media types and display their optimized values
            for mediaName, omediaCol in optimMediaCol1:
                with omediaCol:
                    if mediaName in optimMedia.keys():
                        # Display optimized allocation as percentage or amount based on usePerc flag
                        if usePerc:
                            st.number_input(
                                f"Opt. {mediaName} in %",
                                value=optimMedia[mediaName]["Allocation"] * 100,
                                disabled=True,
                            )
                        else:
                            st.number_input(
                                f"Opt. {mediaName} in €",
                                value=optimMedia[mediaName]["Amount"],
                                disabled=True,
                            )
                    else:
                        # Display zero value if media type is not found in optimMedia
                        if usePerc:
                            st.number_input(
                                f"Opt. {mediaName} in %", value=0, disabled=True
                            )
                        else:
                            st.number_input(
                                f"Opt. {mediaName} in €", value=0, disabled=True
                            )

        # Second container for media allocation display (next 4 media types)
        with st.container():
            optimMediaCol2 = st.columns(4)
            optimMediaCol2 = list(zip(toDisplayMedia[4:], optimMediaCol2))

            # Iterate through the next 4 media types and display their optimized values
            for mediaName, omediaCol in optimMediaCol2:
                with omediaCol:
                    if mediaName in optimMedia.keys():
                        if usePerc:
                            st.number_input(
                                f"Opt. {mediaName} in %",
                                value=optimMedia[mediaName]["Allocation"] * 100,
                                disabled=True,
                            )
                        else:
                            st.number_input(
                                f"Opt. {mediaName} in €",
                                value=optimMedia[mediaName]["Amount"],
                                disabled=True,
                            )
                    else:
                        if usePerc:
                            st.number_input(
                                f"Opt. {mediaName} in %", value=0, disabled=True
                            )
                        else:
                            st.number_input(
                                f"Opt. {mediaName} in €", value=0, disabled=True
                            )

    # Display the optimized ROI in a styled container
    with optimROI:
        with st.container():
            st.empty()
        with st.container():
            st.write("")
            st.markdown(
                f"""
                <div style="
                    color: #070996;
                    width: 100%;
                    text-align: center; 
                    border: 2px solid black; 
                    padding: 0px; 
                    border-radius: 10px; 
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                ">
                    <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">ROI</span>
                    <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">{optimizedROI:.2f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Display the evolution of ROI in a styled container
    with evolROICol:
        with st.container():
            st.empty()
        with st.container():
            st.write("")
            st.markdown(
                f"""
                <div style="
                    color: #070996;
                    width: 100%;
                    text-align: center; 
                    border: 2px solid black; 
                    padding: 0px; 
                    border-radius: 10px; 
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                ">
                    <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">Effectiveness</span>
                    <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">{(evolInitOptim*100):.2f} %</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Display the incremental value in a styled container
    with incremColVal:
        with st.container():
            st.empty()
        with st.container():
            st.write("")
            st.markdown(
                f"""
                <div style="
                    color: #070996;
                    width: 100%;
                    text-align: center; 
                    border: 2px solid black; 
                    padding: 0px; 
                    border-radius: 10px; 
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                ">
                    <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">Incremental</span>
                    <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">{format_currency(incremVal)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Section for the simulator
    st.title("Simulateur Mix Media")
    optimCol, _, _ = st.columns([3, 5, 3])

    # Toggle switch for using optimized budget
    with optimCol:
        useOptimized = st.toggle("Use optimized budget")

    # User input for total budget based on whether optimized budget is used
    with st.container():
        simBudg, _ = st.columns([8, 8])
        with simBudg:
            if useOptimized:
                if "totalBudget" not in st.session_state.keys():
                    totalBudget = st.number_input(
                        "Total budget to simulate",
                        value=optimized_budget["Amount"].sum(),
                    )
                else:
                    totalBudget = st.number_input(
                        "Total budget to simulate", value=st.session_state.totalBudget
                    )
                start_amount = get_budget(totalBudget, optimized_budget)
                grouped_start_amount = (
                    start_amount.drop(["Media Channel", "Amount"], axis=1)
                    .groupby("Media Type")
                    .sum()
                    .reset_index()
                )
            else:
                if "totalBudget" not in st.session_state.keys():
                    totalBudget = st.number_input(
                        "Total budget to simulate",
                        value=previous_budget["Amount"].sum(),
                    )
                else:
                    totalBudget = st.number_input(
                        "Total budget to simulate", value=st.session_state.totalBudget
                    )
                start_amount = get_budget(totalBudget, previous_budget)
                grouped_start_amount = (
                    start_amount.drop(["Media Channel", "Budget", "Amount"], axis=1)
                    .groupby("Media Type")
                    .sum()
                    .reset_index()
                )

    # Columns to display simulated allocation and values for ROI, effectiveness, and incremental value
    with st.container():
        simAlloc, simROI, evolSimROI, simColVal = st.columns([10, 2, 3, 3])

        # Logic for displaying and calculating allocation for media types
        with simAlloc:
            media_cols_pairs1 = list(zip(toDisplayMedia[:4], st.columns(4)))
            mediaCol1 = {e[0]: {"col": e[1]} for e in media_cols_pairs1}

            media_cols_pairs2 = list(zip(toDisplayMedia[4:], st.columns(4)))
            mediaCol2 = {e[0]: {"col": e[1]} for e in media_cols_pairs2}

            # Handle first 4 media types
            with st.container():
                for media, elements in mediaCol1.items():
                    with elements["col"]:
                        if media in list(grouped_start_amount["Media Type"]):
                            if usePerc:
                                elements["input"] = st.number_input(
                                    f"{media} in %",
                                    value=access_type_budget(
                                        grouped_start_amount, media, "Allocation"
                                    ),
                                )
                                elements["data"] = get_channel_allocation(
                                    start_amount,
                                    elements["input"] * totalBudget / 100,
                                    media,
                                )
                            else:
                                elements["input"] = st.number_input(
                                    f"{media} in Euros",
                                    value=access_type_budget(
                                        grouped_start_amount, media, "New Amount"
                                    ),
                                )
                                elements["data"] = get_channel_allocation(
                                    start_amount, elements["input"], media
                                ).reset_index(drop=True)
                        else:
                            if usePerc:
                                elements["input"] = st.number_input(
                                    f"{media} in %", value=0, disabled=True
                                )
                            else:
                                elements["input"] = st.number_input(
                                    f"{media} in Euros", value=0, disabled=True
                                )

            # Handle remaining media types
            with st.container():
                for media, elements in mediaCol2.items():
                    with elements["col"]:
                        if media in list(grouped_start_amount["Media Type"]):
                            if usePerc:
                                elements["input"] = st.number_input(
                                    f"{media} in %",
                                    value=access_type_budget(
                                        grouped_start_amount, media, "Allocation"
                                    ),
                                )
                                elements["data"] = get_channel_allocation(
                                    start_amount,
                                    elements["input"] * totalBudget / 100,
                                    media,
                                )
                            else:
                                elements["input"] = st.number_input(
                                    f"{media} in Euros",
                                    value=access_type_budget(
                                        grouped_start_amount, media, "New Amount"
                                    ),
                                )
                                elements["data"] = get_channel_allocation(
                                    start_amount, elements["input"], media
                                ).reset_index(drop=True)
                        else:
                            if usePerc:
                                elements["input"] = st.number_input(
                                    f"{media} in %", value=0, disabled=True
                                )
                            else:
                                elements["input"] = st.number_input(
                                    f"{media} in Euros", value=0, disabled=True
                                )

    # Combine media allocation inputs into one dictionary
    mediaCol = {k: v for k, v in list(mediaCol1.items()) + list(mediaCol2.items())}

    # Save the simulation data and update the session state
    saveGlobal, _ = st.columns([4, 1])
    with saveGlobal:
        saveButtonGlobal = st.button("Save budget")

        if saveButtonGlobal:
            st.session_state.simulation_data = {
                k: v["data"] for k, v in mediaCol.items() if "data" in v.keys()
            }
            st.session_state.totalBudget = totalBudget

    # Check if total budget matches media allocation, provide warnings if not
    if usePerc == False:
        currentBudget = round(sum([e["input"] for e in mediaCol.values()]))
        if currentBudget != round(totalBudget):
            st.warning(
                f"Total budget is not equal to media allocations (current total : {round(currentBudget, 2)}). Please check values or switch to percentages"
            )
    else:
        currentAll = round(sum([e["input"] for e in mediaCol.values()]))
        if currentAll != 100:
            st.warning(
                f"Total budget allocation is different from 100% (current total : {round(currentAll, 2)}%). Please check allocation."
            )

    # Initialize simulation data in session state if it does not exist
    if "simulation_data" not in st.session_state.keys():
        st.session_state.simulation_data = {
            k: v["data"] for k, v in mediaCol.items() if "data" in v.keys()
        }

    # Create an expander for advanced budget settings
    budgetExpander = st.expander("Advanced settings")

    with budgetExpander:
        # Create columns for media selection, budget type, and advanced percentage toggle
        colSelect, typeBudget, percAdvanced = st.columns([3, 3, 3])

        with colSelect:
            # Dropdown to select media type from simulation data
            typeSelector = st.selectbox(
                label="Media",
                options=list(st.session_state.simulation_data.keys()),
                on_change=reset_channel_budget(),  # Reset channel budget on change
            )

        with typeBudget:
            # Calculate and display the total allocated budget for the selected media type
            budget_sum = st.session_state.simulation_data[typeSelector][
                "New Amount"
            ].sum()
            st.markdown(
                f"The budget allocated to {typeSelector} is {round(budget_sum, 2)}"
            )

        with percAdvanced:
            # Toggle to use percentage for advanced allocation
            useAdvPerc = st.toggle(
                "Use percentage for advanced allocation", value=usePerc
            )

        # Loop through each media channel and create input fields for budget allocation
        for channel in st.session_state.simulation_data[typeSelector]["Media Channel"]:
            if useAdvPerc:
                # Input for percentage allocation if toggled
                newInput = st.number_input(
                    f"Allocation in % for {channel}",
                    value=access_channel_budget(
                        st.session_state.simulation_data[typeSelector],
                        channel,
                        "Allocation",
                    ),
                )
            else:
                # Input for euro allocation if percentage is not used
                newInput = st.number_input(
                    f"Allocation in euros for {channel}",
                    value=access_channel_budget(
                        st.session_state.simulation_data[typeSelector],
                        channel,
                        "New Amount",
                    ),
                )
            # Update the channel budget in the session state
            st.session_state.channel_budget.update({channel: newInput})

        # Validate budget allocation when not using percentages
        if not useAdvPerc:
            if round(budget_sum) != round(
                sum(st.session_state.channel_budget.values())
            ):
                st.warning(
                    f"Total budget is not equal to media allocations ({round(sum(st.session_state.channel_budget.values()), 2)}). Please check values or switch to percentages"
                )
        else:
            # Validate percentage allocations
            if round(sum(st.session_state.channel_budget.values())) != 100:
                st.warning(
                    f"Total budget allocation is greater than 100% ({round(sum(st.session_state.channel_budget.values()), 2)}). Please check allocation."
                )

        _, saveChannel = st.columns([4, 1])
        with saveChannel:
            # Button to save the channel budget
            saveButtonChannel = st.button("Save channel budget")

            if saveButtonChannel:
                # Save the updated budget information to the simulation data
                st.session_state.simulation_data[typeSelector] = saveTypeBudget(
                    st.session_state.simulation_data[typeSelector],
                    st.session_state.channel_budget,
                    usePerc=useAdvPerc,
                    budget=budget_sum,
                )

    # Columns for download options and simulation
    _, downloadExcelCol, downloadPDFCol, simulateCol = st.columns([6, 2, 2, 2])

    with simulateCol:
        # Button to simulate results
        simulateButton = st.button("Simulate results")
        if simulateButton:
            # Combine simulation data and prepare for results
            st.session_state.finalSimulationData = (
                pd.concat([df for df in st.session_state.simulation_data.values()])
                .reset_index(drop=True)
                .drop(["Amount"], axis=1)[["Media Type", "Media Channel", "New Amount"]]
            )
            st.session_state.finalSimulationData = (
                st.session_state.finalSimulationData.rename(
                    {"New Amount": "Amount"}, axis=1
                )
            )
            st.session_state.finalSimulationData = optimized_budget[
                ["Media Type", "Media Channel"]
            ].merge(
                st.session_state.finalSimulationData,
                on=["Media Type", "Media Channel"],
                how="left",
            )

            # Perform the simulation
            st.session_state.simulationResults = makeSimulation(
                st.session_state.finalSimulationData,
                st.session_state.target_scaler,
                st.session_state.media_scaler,
                st.session_state.model,
            )

            # Add simulated budget and allocation to results
            st.session_state.simulationResults["Budget"] = "Simulated"
            st.session_state.simulationResults["Allocation"] = (
                st.session_state.simulationResults["Amount"]
                / st.session_state.simulationResults["Amount"].sum()
            )

    # Check if simulation results exist for downloading
    if "simulationResults" in st.session_state.keys():
        # Convert results to CSV for download
        csv = convert_df(st.session_state.simulationResults)
        with downloadExcelCol:
            st.download_button(
                label="Download data",
                data=csv,
                file_name="simulation_df.csv",
                mime="text/csv",
            )

        # Create tabs for ROI, Contribution, and Budget analysis
        roiTab, contTab, budTab = st.tabs(["ROI", "Contribution", "Budget"])

        # Concatenate optimized budget, previous budget, and simulation results for comparison
        init_optim_df = pd.concat(
            [optimized_budget, previous_budget, st.session_state.simulationResults]
        ).reset_index(drop=True)

        figs = []
        with budTab:
            # Create bar chart for budget comparison
            comparison_df = init_optim_df[["Media Type", "Budget", "Amount"]]
            comparison_df = (
                comparison_df.groupby(["Media Type", "Budget"]).sum().reset_index()
            )
            comparison_df["Amount"] = (
                comparison_df["Amount"] / 1000
            )  # Convert to thousands
            comparison_df["Values"] = comparison_df["Amount"].apply(
                lambda x: f"{x:,.2f}"
            )
            fig = px.bar(
                comparison_df,
                x="Media Type",
                y="Amount",
                color="Budget",
                text="Values",
                barmode="group",
                labels={"Amount": "Investment (k Euros)"},
                color_discrete_sequence=["#000000", "#070996", "#19ACBF"],
            )
            fig.update_traces(textposition="outside")
            figs.append(fig)
            st.plotly_chart(fig)

        with contTab:
            # Create bar chart for contribution comparison
            comparison_df = init_optim_df[["Media Type", "Budget", "Contribution"]]
            comparison_df = (
                comparison_df.groupby(["Media Type", "Budget"]).sum().reset_index()
            )
            comparison_df["Contribution"] = (
                comparison_df["Contribution"] / 1000
            )  # Convert to thousands
            comparison_df["Values"] = comparison_df["Contribution"].apply(
                lambda x: f"{x:,.2f}"
            )
            fig = px.bar(
                comparison_df,
                x="Media Type",
                y="Contribution",
                color="Budget",
                text="Values",
                barmode="group",
                labels={"Contribution": "Contribution (k Euros)"},
                color_discrete_sequence=["#000000", "#070996", "#19ACBF"],
            )
            fig.update_traces(textposition="outside")
            figs.append(fig)
            st.plotly_chart(fig)

        with roiTab:
            # Create bar chart for ROI comparison
            comparison_df = init_optim_df[
                ["Media Type", "Budget", "Amount", "Contribution"]
            ]
            comparison_df = (
                comparison_df.groupby(["Media Type", "Budget"]).sum().reset_index()
            )
            comparison_df["ROI"] = (
                comparison_df["Contribution"] / comparison_df["Amount"]
            )
            comparison_df["Values"] = comparison_df["ROI"].apply(lambda x: f"{x:,.2f}")
            fig = px.bar(
                comparison_df,
                x="Media Type",
                y="ROI",
                color="Budget",
                text="Values",
                barmode="group",
                color_discrete_sequence=["#000000", "#070996", "#19ACBF"],
            )
            fig.update_traces(textposition="outside")
            figs.append(fig)
            st.plotly_chart(fig)

        # Prepare images for PDF report
        image_files = ["plot1.png", "plot2.png", "plot3.png"]
        image_files = [
            "projets/" + session_params["mission"] + "/" + f for f in image_files
        ]

        for fig, filename in zip(figs, image_files):
            fig.write_image(filename)

        # Function to create PDF report from images
        def make_pdf(image_files, output_filename):
            pdf = FPDF(orientation="L", unit="mm", format="A4")
            pdf.set_auto_page_break(0)
            for image_file in image_files:
                pdf.add_page()
                pdf.image(image_file, x=10, y=10, w=277)
            pdf.output(output_filename)

        pdf_filename = (
            "projets/" + session_params["mission"] + "/simulateur_rapport.pdf"
        )
        make_pdf(image_files, pdf_filename)

        # Read PDF file for download
        with open(
            "projets/" + session_params["mission"] + "/simulateur_rapport.pdf", "rb"
        ) as pdf_file:
            document = pdf_file.read()

        with downloadPDFCol:
            st.download_button(
                label="Download PDF",
                data=document,
                file_name="simulateur_rapport.pdf",
                mime="application/pdf",
            )

        # Calculate ROI and incremental values from the simulation results
        roi_simulation = sum(st.session_state.simulationResults["Contribution"]) / sum(
            st.session_state.simulationResults["Amount"]
        )
        roi_evol_simulation_perc = (roi_simulation / initialROI) - 1
        increm_simulation_val = sum(
            st.session_state.simulationResults["Contribution"]
        ) - sum(grouped_previous["Contribution"])

        # Display ROI simulation results
        with simROI:
            with st.container():
                st.empty()
            with st.container():
                st.write("")
                st.markdown(
                    f"""
                        <div style="
                            color: #19acbf;
                            width: 100%;
                            text-align: center; 
                            border: 2px solid black; 
                            padding: 0px; 
                            border-radius: 10px; 
                            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                        ">
                            <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">ROI</span>
                            <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">{roi_simulation:.2f}</span>
                        </div>
                        """,
                    unsafe_allow_html=True,
                )

        # Display effectiveness simulation results
        with evolSimROI:
            with st.container():
                st.empty()
            with st.container():
                st.write("")
                st.markdown(
                    f"""
                        <div style="
                            color: #19acbf;
                            width: 100%;
                            text-align: center; 
                            border: 2px solid black; 
                            padding: 0px; 
                            border-radius: 10px; 
                            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                        ">
                            <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">Effectiveness</span>
                            <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">{(roi_evol_simulation_perc*100):.2f} (%)</span>
                        </div>
                        """,
                    unsafe_allow_html=True,
                )

        # Display incremental simulation value results
        with simColVal:
            with st.container():
                st.empty()
            with st.container():
                st.write("")
                st.markdown(
                    f"""
                        <div style="
                            color: #19acbf;
                            width: 100%;
                            text-align: center; 
                            border: 2px solid black; 
                            padding: 0px; 
                            border-radius: 10px; 
                            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                        ">
                            <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">Incremental</span>
                            <span style="display: block; font-size: 20px; font-weight: bold; margin: 0; padding: 10px 0;">{format_currency(increm_simulation_val)}</span>
                        </div>
                        """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
