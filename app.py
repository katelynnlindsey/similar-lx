import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean

# Set page configuration for a better appearance
st.set_page_config(page_title="Linguistics Department Similarity", layout="wide")

# Link to the other app
st.markdown(
    "[Switch to Categorize Linguistics Departments as Peers/Aspirant Peers App](https://categorize-lx.streamlit.app)",
    unsafe_allow_html=True
)

# Add this to make the charts more color-blind friendly
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00"])

# Load the data
data = pd.read_csv('universities_data.csv')

# Keep a copy of the original data for charting
original_data = data.copy()

# Define the list of all subfields (only if you are using subfield representations)
all_subfields = [
    "syntax", "semantics", "acquisition", "sociolinguistics", "applied",
    "morphology", "phonetics", "general", "psycholinguistics", "computational",
    "pragmatics", "historical", "documentation", "neurolinguistics",
    "sign language", "processing", "cognition", "philosophy", "bilingualism",
    "typology", "tesol", "cogneurosci", "discourse", "vision", "evolution",
    "pathology", "psychology", "translation", "computer science", "semiotics",
    "deaf education", "women studies", "literature", "corpus", "variation",
    "heritage", "disorders", "anthropology", "fieldwork", "game theory",
    "digital humanities"
]

# Define a mapping for each feature to its column(s)
feature_mapping = {
    'Number of Faculty (2019)': 'Number of Faculty (2019)',
    'Number of Graduate Students (2019)': 'Number of Graduate Students (2019)',
    'Faculty Ranking (2024)': 'Faculty Ranking (2024)',
    'Age of Faculty Dissertations (2019)': 'Age of Faculty Dissertations (2019)',
    'Average h-index/age of Faculty (2024)': 'Average h-index/age of Faculty (2024)',
    'Average h-index/age of Graduates w TT/T jobs in NA (2024)': 'Average h-index/age of Graduates w TT/T jobs in NA (2024)',
    'Gender Ratio: Men/Women (2019)': 'Gender Ratio: Men/Women (2019)',
    'Average number of LX Majors (2019-2023)': 'Average number of LX Majors (2019-2023)',
    'Subfield representation (2019)': list(range(len(all_subfields)))  # Indices corresponding to subfield representation
}

# Normalize and preprocess the selected features
def preprocess_data(data, selected_features):
    # Separate subfield processing from other features
    if 'Subfield representation (2019)' in selected_features:
        subfield_vectors = data['Subfield representation (2019)'].apply(preprocess_subfields).tolist()
        subfield_df = pd.DataFrame(subfield_vectors, columns=all_subfields)
        data = pd.concat([data, subfield_df], axis=1)
        selected_features.remove('Subfield representation (2019)')
        selected_features.extend(all_subfields)
    
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

    # Define colors for highlighting
    color_palette = ["#ff7f00", "#ffd700", "#4daf4a", "#377eb8"]  # Orange, Yellow, Green, Blue
    highlight_color = "#d73027"  # Red for the selected university
    neutral_color = "#cccccc"  # Gray for others

    # Assign colors and labels for the top similar departments
    rank_labels = ['1st', '2nd', '3rd', '4th']
    similar_dept_colors = {
        university: (color_palette[i], rank_labels[i]) if i < len(color_palette) else (neutral_color, f"{i+1}th")
        for i, (university, _) in enumerate(top_similar)
    }
    
    # Highlight the selected university distinctly in red with a label of 0
    similar_dept_colors[selected_university] = (highlight_color, '0')

    # Display the results
    st.write(f"Top {num_similar_departments} linguistics departments most similar to {selected_university} based on selected features:")
    comparison_table = pd.DataFrame(top_similar, columns=['University Name', 'Euclidean Distance'])
    st.dataframe(comparison_table)

    # Create a tabbed interface for charts
st.write("### Attribute Comparisons")
# Filter out "Subfield representation (2019)" from tab_names
tab_names = [attr for attr in selected_features if attr not in all_subfields]
if tab_names:
    tabs = st.tabs(tab_names)
    for tab, attribute in zip(tabs, tab_names):
        with tab:
            sorted_data = original_data.sort_values(by=attribute, ascending=False)
            num_universities = len(sorted_data)
            plt.figure(figsize=(10, num_universities * 0.4))  # Increase height dynamically

            # Extract colors and labels for the bars
            bar_colors = [
                similar_dept_colors.get(univ, (neutral_color, ''))[0] 
                for univ in sorted_data['University Name']
            ]
            bar_labels = [
                similar_dept_colors.get(univ, (neutral_color, ''))[1]
                for univ in sorted_data['University Name']
            ]

            bars = plt.barh(
                sorted_data['University Name'], sorted_data[attribute], color=bar_colors
            )
            
            # Add value labels to the bars (both the ranking numbers and the attribute value)
            for bar, label in zip(bars, bar_labels):
                plt.text(
                    bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f'{label}',
                    va='center',
                    ha='center',
                    weight='bold',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
                )
                plt.text(
                    bar.get_width() + 0.25,
                    bar.get_y() + bar.get_height() / 2,
                    f'{bar.get_width():.2f}',
                    va='center',
                    ha='left'
                )
            
            plt.xlabel(attribute)
            plt.ylabel('University Name')
            plt.title(f'{attribute} Comparison')
            plt.xlim(0, sorted_data[attribute].max() * 1.1)
            
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
