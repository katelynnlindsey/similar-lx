import streamlit as st
import pandas as pd

st.markdown(
    "[Switch to Similarity Comparison App](https://similar-lx.streamlit.app)",
    unsafe_allow_html=True
)

# Load the CSV data
data = pd.read_csv('universities_data.csv')

# Define categories for selection
categories = ["Selected University", "Peers", "Aspirant Peers", "Neither"]

# Initialize session state for selections
if 'selections' not in st.session_state:
    st.session_state.selections = {category: [] for category in categories}

# Function to update selections and ensure mutual exclusivity
def update_selections(selected_unis, category):
    # Update the current category with the new selections
    st.session_state.selections[category] = selected_unis
    
    # Ensure exclusivity: Remove selected universities from other categories
    for cat in categories:
        if cat != category:
            st.session_state.selections[cat] = [
                uni for uni in st.session_state.selections[cat] if uni not in selected_unis
            ]

# Title of the app
st.title("University Categorization App")

# Selection Interface
st.subheader("Categorize Universities")
st.write("Select universities for each category. 'Selected University' allows only one selection, while others can have multiple.")

# Get the list of university names
university_names = data['University Name'].tolist()

# Create selection widgets for each category
selected_uni = st.selectbox(
    "Selected University (Only One)", 
    options=["None"] + university_names, 
    index=0 if not st.session_state.selections["Selected University"] else university_names.index(st.session_state.selections["Selected University"][0]) + 1
)
if selected_uni != "None":
    update_selections([selected_uni], "Selected University")

# Create multi-select widgets for other categories with exclusivity
for category in ["Peers", "Aspirant Peers", "Neither"]:
    available_options = [
        uni for uni in university_names if uni not in st.session_state.selections["Selected University"]
        and uni not in sum((st.session_state.selections[cat] for cat in categories if cat != category), [])
    ]
    
    selected_unis = st.multiselect(
        f"{category} (Select multiple)",
        options=available_options,
        default=st.session_state.selections[category]
    )
    # Directly update the state based on the user's selections
    update_selections(selected_unis, category)
    # Display selected universities below each multi-select
    st.write(f"Selected in {category}: {', '.join(st.session_state.selections[category]) or 'None'}")

# Function to calculate the summary for a selected category
def calculate_summary(selected_universities, df, include_min_max=True):
    if not selected_universities:
        return pd.Series({"Summary": "None"})
    
    selected_data = df[df['University Name'].isin(selected_universities)]
    numeric_cols = selected_data.select_dtypes(include=['number']).columns
    
    if include_min_max:
        summary = selected_data[numeric_cols].apply(
            lambda col: f"{col.mean():.2f} ({col.min():.2f} - {col.max():.2f})"
        )
    else:
        summary = selected_data[numeric_cols].mean().round(2)
    
    return summary

# Create an empty DataFrame for displaying the summary
summary_df = pd.DataFrame()

# Collect summaries for all categories
summary_df["Selected University"] = calculate_summary(
    st.session_state.selections["Selected University"], data, include_min_max=False
)
for category in ["Peers", "Aspirant Peers", "Neither"]:
    summary_df[category] = calculate_summary(
        st.session_state.selections[category], data, include_min_max=True
    )

# Display the unified summary table
st.write("### Summary Table")
st.dataframe(summary_df, height=400, use_container_width=True)

# Prepare data for export
export_data = summary_df.copy()
# Add a row for the selected universities in each field
for category in categories:
    export_data.loc['Selected Universities', category] = ', '.join(st.session_state.selections[category])

# Provide download link for the CSV file
csv_export = export_data.to_csv(index=True)
st.download_button(
    label="Download Summary as CSV",
    data=csv_export,
    file_name="university_summary.csv",
    mime="text/csv"
)
