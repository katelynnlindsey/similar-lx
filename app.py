import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean

# Set page configuration for a better appearance
st.set_page_config(page_title="Linguistics Department Similarity", layout="wide")

# Add this to make the charts more color-blind friendly
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00"])

# Load the data
data = pd.read_csv('universities_data.csv')

# List of colors for highlighting
highlight_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3']

# Keep a copy of the original data for charting
original_data = data.copy()

# Normalize and preprocess the selected features
def preprocess_data(data, selected_features):
    # Replace NaNs with zeros before normalization to avoid issues
    data[selected_features] = data[selected_features].fillna(0)
    
    # Normalize the selected features
    scaler = MinMaxScaler()
    data[selected_features] = scaler.fit_transform(data[selected_features])
    
    return data

# Function to preprocess subfields into a vector
def preprocess_subfields(subfield_str):
    subfield_dict = {subfield: 0 for subfield in all_subfields}
    if pd.isna(subfield_str):
        return list(subfield_dict.values())

    subfields = subfield_str.split(", ")
    for key in subfields:
        try:
            count, name = key.split(" ", 1)
            count = int(count.strip().replace('"', ''))
            subfield_dict[name] = count
        except (ValueError, IndexError):
            continue
    return [subfield_dict[subfield] for subfield in all_subfields]

# UI for user input
st.title("Linguistics Department Similarity Calculator")
selected_university = st.selectbox("Select a linguistics department:", data['University Name'])

# Add description below the dropdown but above the checkboxes
st.write("Select the attributes you want to use to measure similarity between the linguistics departments.")

# Checkboxes for attribute selection
selected_features = []
if st.checkbox("Number of Faculty (2019)"):
    selected_features.append('Number of Faculty (2019)')
if st.checkbox("Number of Graduate Students (2019)"):
    selected_features.append('Number of Graduate Students (2019)')
if st.checkbox("Faculty Ranking (2024)"):
    selected_features.append('Faculty Ranking (2024)')
if st.checkbox("Age of Faculty Dissertations (2019)"):
    selected_features.append('Age of Faculty Dissertations (2019)')
if st.checkbox("Average h-index/age of Faculty (2024)"):
    selected_features.append('Average h-index/age of Faculty (2024)')
if st.checkbox("Average h-index/age of Graduates w TT/T jobs in NA (2024)"):
    selected_features.append('Average h-index/age of Graduates w TT/T jobs in NA (2024)')
if st.checkbox("Gender Ratio: Men/Women (2019)"):
    selected_features.append('Gender Ratio: Men/Women (2019)')
if st.checkbox("Average number of LX Majors (2019-2023)"):
    selected_features.append('Average number of LX Majors (2019-2023)')
if st.checkbox("Subfield representation (2019)"):
    selected_features.append('Subfield representation (2019)')

# Input for the number of similar departments to show
num_similar_departments = st.number_input(
    "Number of similar departments to display:", min_value=1, max_value=20, value=5
)

# Button to trigger the similarity calculation
if st.button("Find Similar Linguistics Departments"):
    selected_indices = []
    for feature in selected_features:
        indices = feature_mapping[feature]
        if isinstance(indices, list):
            selected_indices.extend(indices)
        else:
            selected_indices.append(indices)

    # Preprocess the data for normalization and NaN handling
    data = preprocess_data(data, selected_features)
    
    # Retrieve the vector for the selected university
    selected_university_vector = data.loc[data['University Name'] == selected_university, selected_features].values[0]

    # Extract vectors for all universities for comparison
    filtered_vectors = data[selected_features].values

    # Compute Euclidean distances and handle NaNs or Infs
    distances = []
    for vector, university in zip(filtered_vectors, data['University Name']):
        if university != selected_university:
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
            selected_university_vector = np.nan_to_num(selected_university_vector, nan=0.0, posinf=0.0, neginf=0.0)
            dist = euclidean(selected_university_vector, vector)
            distances.append((university, dist))

    # Sort by distance (smaller is more similar)
    distances.sort(key=lambda x: x[1])
    top_similar = distances[:num_similar_departments]

    # Assign colors to similar departments
    similar_dept_colors = {}
    for i, (university, _) in enumerate(top_similar):
        color = highlight_colors[i % len(highlight_colors)]
        similar_dept_colors[university] = color

    # Display the results
    st.write(f"Top {num_similar_departments} linguistics departments most similar to {selected_university} based on selected features:")
    comparison_table = pd.DataFrame(top_similar, columns=['University Name', 'Euclidean Distance'])
    st.dataframe(comparison_table)

    # Create a tabbed interface for charts
    st.write("### Attribute Comparisons")
    tab_names = [attr for attr in selected_features if attr != 'Subfield representation (2019)']
    if tab_names:
        tabs = st.tabs(tab_names)
        for tab, attribute in zip(tabs, tab_names):
            with tab:
                sorted_data = original_data.sort_values(by=attribute, ascending=False)
                num_universities = len(sorted_data)
                plt.figure(figsize=(10, num_universities * 0.4))  # Increase height dynamically

                bars = plt.barh(sorted_data['University Name'], sorted_data[attribute], color=[
                    similar_dept_colors.get(univ, "#999999") for univ in sorted_data['University Name']
                ])
                
                # Add the selected university with a distinct color
                selected_bar = plt.barh(
                    selected_university, 
                    sorted_data.loc[sorted_data['University Name'] == selected_university, attribute], 
                    color="#377eb8"
                )
                
                plt.xlabel(attribute)
                plt.ylabel('University Name')
                plt.title(f'{attribute} Comparison')
                plt.xlim(0, sorted_data[attribute].max() * 1.1)

                # Add value labels to the bars
                for bar in bars:
                    plt.text(
                        bar.get_width() + 0.05,
                        bar.get_y() + bar.get_height() / 2,
                        f'{bar.get_width():.2f}',
                        va='center',
                        ha='left'
                    )
                for bar in selected_bar:
                    plt.text(
                        bar.get_width() + 0.05,
                        bar.get_y() + bar.get_height() / 2,
                        f'{bar.get_width():.2f}',
                        va='center',
                        ha='left'
                    )
                
                st.pyplot(plt)


# Add data sources at the bottom
st.write("### Data Sources")
st.markdown("""
Number of Faculty (2019) comes from Haugen et al. (2024). [^1] 
Number of Graduate Students (2019) comes from Petersons. [^2]  
Faculty Ranking (2024) comes from a private survey of 8 faculty members.  
Age of Faculty Dissertations (2019) comes from Haugen et al. (2024).[^1]  
Average h-index/age of Faculty comes from Haugen et al. (2024) and Google Scholar Profiles (2024).[^1][^3]  
Average h-index/age of graduates comes from Haugen et al. (2024) and Google Scholar Profiles (2024).[^1][^3]  
Gender ratio comes from the Bias in Linguistics project.[^4]  
Average number of LX majors comes from the IPEDS data center.[^5]  
Subfield representation comes from Haugen et al. (2024).[^1]  

[^1]: Haugen, J. D., Margaris, A. V., & Calvo, S. E. (2024). A Snapshot of Academic Job Placements in Linguistics in the US and Canada. Canadian Journal of Linguistics/Revue canadienne de linguistique, 69(1), 129-143.  
[^2]: [Petersons](https://www.petersons.com/search/grad?q=linguistics).  
[^3]: Google Scholar Profiles (2024).  
[^4]: [Bias in Linguistics project](https://biasinlinguistics.org/).  
[^5]: [IPEDS data center](https://nces.ed.gov/ipeds/use-the-data).  
""")
