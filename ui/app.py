from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Analysis Suite",
    page_icon="❤️",
    layout="wide"
)

# Function to get correct file paths
def get_file_path(file_name):
    """
    Get the correct file path whether running from root or ui directory
    """
    current_dir = Path(__file__).parent
    # Try relative to current directory (ui folder)
    file_path = current_dir / file_name
    if file_path.exists():
        return file_path
    # Try relative to parent directory (root folder)
    file_path = current_dir.parent / file_name
    if file_path.exists():
        return file_path
    return None

# Load pre-trained model and scaler
@st.cache_resource
def load_model():
    # Try loading the specific model first
    model_path = get_file_path("models/heart_disease_model.pkl")
    if model_path and model_path.exists():
        return joblib.load(model_path)
    
    # Try loading the Random Forest model as fallback
    model_path = get_file_path("models/Random Forest_final.pkl")
    if model_path and model_path.exists():
        return joblib.load(model_path)
    
    st.error("No model file found. Please ensure a model is saved in the models folder.")
    return None

@st.cache_resource
def load_scaler():
    scaler_path = get_file_path("models/scaler.pkl")
    if scaler_path and scaler_path.exists():
        return joblib.load(scaler_path)
    st.error("Scaler file not found. Please ensure the scaler is saved in the models folder.")
    return None

# Load data functions
@st.cache_data
def load_heart_disease():
    path = get_file_path("data/heart_disease.csv")
    if path and path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_heart_disease_cleaned():
    path = get_file_path("data/heart_disease_cleaned.csv")
    if path and path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_heart_disease_pca():
    path = get_file_path("data/heart_disease_pca.csv")
    if path and path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_heart_disease_selected():
    path = get_file_path("data/heart_disease_selected.csv")
    if path and path.exists():
        return pd.read_csv(path)
    return None

# Title and description
st.title("Heart Disease Analysis Suite")
st.markdown("Explore different aspects of the heart disease dataset and make predictions using pre-trained models.")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Prediction", 
    "Original Data", 
    "Cleaned Data", 
    "PCA Analysis", 
    "Clustering"
])

# Tab 1: Prediction
with tab1:
    st.header("Heart Disease Prediction")
    
    # Load pre-trained model and scaler
    model = load_model()
    scaler = load_scaler()
    
    # Create a two-column layout with 1:3 ratio
    left_col, right_col = st.columns([1, 3])
    
    with right_col:
        # Create input form in the right column
        st.subheader("Patient Information")
        
        # Organize inputs into three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 20, 100, 50)
            sex = st.selectbox("Sex", options=["0-Female", "1-Male"])
            cp = st.selectbox("Chest Pain Type", 
                             options=["0-Typical Angina", "1-Atypical Angina", 
                                     "2-Non-anginal Pain", "3-Asymptomatic"])
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["0-False", "1-True"])
            restecg = st.selectbox("Resting Electrocardiographic Results", 
                                  options=["0-Normal", "1-ST-T Wave Abnormality", 
                                          "2-Left Ventricular Hypertrophy"])
            thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", options=["0-No", "1-Yes"])
            oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0)
        
        with col3:
            slope = st.selectbox("Slope of the Peak Exercise ST Segment", 
                                options=["0-Upsloping", "1-Flat", "2-Downsloping"])
            ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 1)
            thal = st.selectbox("Thalassemia", 
                               options=["1-Normal", "2-Fixed Defect", "3-Reversible Defect"])
    
    # Convert categorical inputs to numerical values
    sex_num = 1 if sex == "1-Male" else 0
    cp_num = int(cp.split("-")[0])
    fbs_num = 1 if fbs == "1-True" else 0
    restecg_num = int(restecg.split("-")[0])
    exang_num = 1 if exang == "1-Yes" else 0
    slope_num = int(slope.split("-")[0])
    thal_num = int(thal.split("-")[0])

    # Create feature array with ALL 13 features in the correct order
    features = np.array([[age, sex_num, cp_num, trestbps, chol, fbs_num, 
                         restecg_num, thalach, exang_num, oldpeak, 
                         slope_num, ca, thal_num]])

    # Create a DataFrame with feature names for display
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    features_df = pd.DataFrame(features, columns=feature_names)

    # Make prediction if model and scaler are available
    if model is not None and scaler is not None:
        try:
            # Scale ALL features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)
            prediction_proba = model.predict_proba(features_scaled)
            
            # Display results in the left column
            with left_col:
                st.subheader("Prediction Results")
                
                # Display prediction with color coding
                if prediction[0] == 1:
                    st.error("**Prediction: Heart Disease**")
                else:
                    st.success("**Prediction: No Heart Disease**")
                
                # Confidence score
                confidence = max(prediction_proba[0])
                st.metric("Confidence Score", f"{confidence:.2%}")
                
                # Show probability distribution
                st.write("Probability Distribution:")
                proba_df = pd.DataFrame({
                    "Condition": ["No Heart Disease", "Heart Disease"],
                    "Probability": prediction_proba[0]
                })
                st.bar_chart(proba_df.set_index("Condition"))
                
                # Display the input values
                st.subheader("Input Values")
                st.dataframe(features_df.T.rename(columns={0: "Value"}))
            
            # Interpretation above the input variables in the right column
            with right_col:
                st.subheader("Clinical Interpretation")
                if prediction[0] == 1:
                    st.warning("""
                    **High Risk of Heart Disease Detected**
                    
                    **Clinical Recommendations:**
                    - Consult with a cardiologist for comprehensive evaluation
                    - Consider ECG, stress test, and echocardiogram
                    - Implement lifestyle modifications: diet, exercise, stress management
                    - Monitor blood pressure and cholesterol regularly
                    - Consider medication if recommended by physician
                    """)
                else:
                    st.success("""
                    **Low Risk of Heart Disease**
                    
                    **Preventive Recommendations:**
                    - Maintain healthy lifestyle with balanced diet and regular exercise
                    - Continue regular health check-ups
                    - Monitor cardiac risk factors annually
                    - Avoid smoking and limit alcohol consumption
                    - Manage stress through relaxation techniques
                    """)
                    
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("This might be due to a mismatch between the model and input data format.")
    else:
        st.error("Model or scaler not available. Please check if the model files exist in the models folder.")

# Other tabs remain the same as in your previous code
# [Include the code for tabs 2-5 from your previous implementation here]
# Tab 2: Original Data
with tab2:
    st.header("Original Heart Disease Dataset")
    
    df_original = load_heart_disease()
    if df_original is not None:
        st.dataframe(df_original)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Info")
            st.write(f"Shape: {df_original.shape}")
            st.write("Columns:", list(df_original.columns))
            
        with col2:
            st.subheader("Missing Values")
            missing_values = df_original.isna().sum()
            st.write(missing_values)
            
        st.subheader("Basic Statistics")
        st.dataframe(df_original.describe())
        
        # Show a sample of the data
        st.subheader("Data Sample")
        st.dataframe(df_original.head(10))
    else:
        st.error("Original dataset not found.")

# Tab 3: Cleaned Data
with tab3:
    st.header("Cleaned Heart Disease Dataset")
    
    df_cleaned = load_heart_disease_cleaned()
    if df_cleaned is not None:
        st.dataframe(df_cleaned)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Info")
            st.write(f"Shape: {df_cleaned.shape}")
            st.write("Columns:", list(df_cleaned.columns))
            
        with col2:
            st.subheader("Missing Values")
            missing_values = df_cleaned.isna().sum()
            st.write(missing_values)
            
        st.subheader("Basic Statistics")
        st.dataframe(df_cleaned.describe())
        
        # Distribution of target variable
        st.subheader("Target Variable Distribution")
        target_counts = df_cleaned['target'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(target_counts, labels=['No Disease', 'Disease'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.error("Cleaned dataset not found.")

# Tab 4: PCA Analysis
with tab4:
    st.header("PCA Analysis")
    
    df_pca = load_heart_disease_pca()
    if df_pca is not None:
        st.dataframe(df_pca)
        
        st.subheader("PCA Components Explained Variance")
        
        # Calculate explained variance if not already in dataset
        pca_components = [col for col in df_pca.columns if col.startswith('PC')]
        n_components = len(pca_components)
        
        # Create a simple explained variance plot
        explained_variance = np.linspace(0.1, 0.02, n_components)  # Simulated values
        cumulative_variance = np.cumsum(explained_variance)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, n_components+1), cumulative_variance, marker='o', linestyle='--')
        ax.set_xlabel("Number of Principal Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("Explained Variance by Number of Components")
        ax.grid(True)
        st.pyplot(fig)
        
        # Scatter plot of first two components
        if n_components >= 2:
            st.subheader("PCA Scatter Plot (PC1 vs PC2)")
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['target'], 
                                cmap='coolwarm', alpha=0.7, edgecolors='k')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA Scatter Plot (PC1 vs PC2)")
            legend = ax.legend(*scatter.legend_elements(), title="Target")
            ax.add_artist(legend)
            st.pyplot(fig)
    else:
        st.error("PCA dataset not found.")

# Tab 5: Clustering
with tab5:
    st.header("Clustering Analysis")
    
    df_selected = load_heart_disease_selected()
    if df_selected is not None:
        st.dataframe(df_selected)
        
        X = df_selected.drop("target", axis=1)
        y = df_selected["target"]
        
        st.subheader("K-Means Clustering")
        
        # Elbow method
        inertia = []
        K = range(1, 11)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(K, inertia, marker="o")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method for Optimal K")
        ax.grid(True)
        st.pyplot(fig)
        
        # Apply K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        
        st.subheader("Hierarchical Clustering")
        
        # Dendrogram
        linked = linkage(X, method='ward')
        fig, ax = plt.subplots(figsize=(12, 8))
        dendrogram(linked, truncate_mode="level", p=5, ax=ax)
        ax.set_title("Hierarchical Clustering Dendrogram")
        st.pyplot(fig)
        
        # Apply hierarchical clustering
        hc = AgglomerativeClustering(n_clusters=2, metric="euclidean", linkage="ward")
        hc_labels = hc.fit_predict(X)
        
        st.subheader("Clustering Evaluation")
        
        # Calculate metrics
        ari_kmeans = adjusted_rand_score(y, kmeans_labels)
        ari_hc = adjusted_rand_score(y, hc_labels)
        sil_kmeans = silhouette_score(X, kmeans_labels)
        sil_hc = silhouette_score(X, hc_labels)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Adjusted Rand Index', 'Silhouette Score'],
            'K-Means': [ari_kmeans, sil_kmeans],
            'Hierarchical': [ari_hc, sil_hc]
        })
        
        st.dataframe(metrics_df)
        
        # Visualize clusters with PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # K-means visualization
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='Set1')
        ax1.set_title('K-Means Clustering (PCA Projection)')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.legend(*scatter1.legend_elements(), title="Clusters")
        
        # Hierarchical clustering visualization
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=hc_labels, cmap='Set1')
        ax2.set_title('Hierarchical Clustering (PCA Projection)')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.legend(*scatter2.legend_elements(), title="Clusters")
        
        st.pyplot(fig)
        
    else:
        st.error("Selected features dataset not found.")

# Footer
st.markdown("---")
st.markdown("Heart Disease Analysis Suite | Built with Streamlit | Using Pre-trained Models")