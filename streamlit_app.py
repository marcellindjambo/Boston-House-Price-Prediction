import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error

# Set Streamlit page configuration
st.set_page_config(
    page_title="House Price Prediction App",
    page_icon="🏠",
    layout="wide",
)

st.write("""
# Boston House Price Prediction App         

The purpose of this project is to develop a machine learning model to predict housing prices in the city of Boston. 
The project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology**,
which is a widely used approach for data mining and machine learning projects.

""")

st.write("---")

# Load the data
@st.cache_resource
def load_data(file_path):
    return pd.read_csv(file_path)


data_file = r"https://raw.githubusercontent.com/marcellindjambo/Boston-House-Price-Prediction/main/data/housing_data.csv"
data = load_data(data_file)
X = data.drop('MedianHomeValue', axis=1)
y = data['MedianHomeValue']

# Sidebar
# Specify input parameters
st.sidebar.header('Specify Input Parameters')


# Define the list of features along with their descriptions
features_info = {
    'PerCapCrimeRate': 'Per Capita Crime Rate',
    'ResidentialLandOver25K': 'Proportion of Residential Land Over 25,000 sq. ft.',
    'NonRetailAcresRatio': 'Proportion of Non-Retail Business Acres',
    'CharlesRiverDummy': 'Proximity to Charles River (1 if tract bounds river; 0 otherwise)',
    'NitricOxideConc': 'Nitric Oxides Concentration (parts per 10 million)',
    'AvgNumRooms': 'Average Number of Rooms per Dwelling',
    'Pre1940sOwnerOcc': 'Proportion of Owner-Occupied Units Built Before 1940',
    'WeightedDistToEmploy': 'Weighted Distance to Employment Centers',
    'RadialHighwayAccessIndex': 'Accessibility to Radial Highways',
    'MedianHomeTaxRate': 'Median Property Tax Rate (per $10,000)',
    'PupilTeacherRatio': 'Pupil-Teacher Ratio by Town',
    'AfroAmericanProportion': 'Proportion of Afro-American Population',
    'LowerStatusRatio': 'Proportion of Population with Lower Status'
}

# Create sliders for each feature in the sidebar
user_inputs = {}
for feature, description in features_info.items():
    min_val = X[feature].min()
    max_val = X[feature].max()
    default_val = X[feature].mean()
    user_inputs[feature] = st.sidebar.slider(f"{description}", float(min_val), float(max_val), float(default_val))

# Create a dictionary to hold the user inputs
data = {feature: user_inputs[feature] for feature in features_info.keys()}

# Create a DataFrame from the user inputs
df = pd.DataFrame(data, index=[0])

# Main panel
# Print the specified input parameters
st.header("Specified Input Parameters")
st.write(df)
st.write("---")

# Load the trained model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model_path = r"model_saved/random_forest_model_0.pkl"
model = load_model(model_path)

# Prediction
y_pred = model.predict(df)

# Model Evaluation Metrics
score = model.score(X_test, y_test)
mse = mean_squared_error(y_test, model.predict(X_test))
mae = mean_absolute_error(y_test, model.predict(X_test))
max_err = max_error(y_test, model.predict(X_test))

# Display prediction and score
st.header("Prediction of Median Home Value")
st.write(y_pred)
st.write('---')

# Display model evaluation metrics in columns
st.header("Model Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("R-squared")
    st.write(score)

with col2:
    st.write("Mean Squared Error")
    st.write(mse)

with col3:
    st.write("Mean Absolute Error")
    st.write(mae)

with col4:
    st.write("Max Error" )
    st.write(max_err)

st.write('---')

# Explaining the model's predictions using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df)

# Feature Importance
st.header("Feature Importance")

with st.expander("Feature Importance Based on Shap values"):
    st.pyplot(
        shap.summary_plot(shap_values, df),
        use_container_width=True,
        bbox_inches='tight',
    )
st.write('---')

# Plot SHAP summary plot as bar chart
with st.expander("Feature Importance Based on SHAP values (Bar)"):
    st.pyplot(
        shap.summary_plot(shap_values, df, plot_type='bar'),
        use_container_width=True,
        bbox_inches='tight',
    )
st.write('---')

st.markdown(
    """
    ### Author Information

    #### Marcellin Djambo
    _Junior Data Scientist_

    [LinkedIn](https://www.linkedin.com/in/marcellindjambo)
    """
)
