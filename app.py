import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation using K-Means",
    page_icon="🛍️",
    layout="wide"
)

# Title and Introduction
st.title("🛍️ Customer Segmentation using K-Means Clustering")
st.markdown("---")
st.markdown("""
### Welcome to the Customer Segmentation App!
This application helps you segment customers into different groups based on their characteristics using the K-Means clustering algorithm.
""")

# Load the dataset
@st.cache_data
def load_data():
    """Load the customer dataset from the specified path."""
    dataset_path = "C:/Users/kumar/OneDrive/Desktop/streamlit/internship/Prodigy_info/task/task_2/Mall_Customers.csv"
    try:
        df = pd.read_csv(dataset_path)
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found at: {dataset_path}")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load data
df = load_data()

if df is not None:
    # Sidebar for controls
    st.sidebar.header("⚙️ Clustering Parameters")
    
    # Display dataset info
    st.header("📊 Dataset Overview")
    st.write(f"**Total Customers:** {len(df)}")
    st.write(f"**Total Features:** {len(df.columns)}")
    
    # Display original dataset
    st.subheader("Original Dataset")
    st.dataframe(df, use_container_width=True)
    
    # Data Preprocessing Section
    st.markdown("---")
    st.header("🔧 Data Preprocessing")
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'CustomerID' in numerical_cols:
        numerical_cols.remove('CustomerID')
    
    st.write(f"**Numerical columns selected for clustering:** {', '.join(numerical_cols)}")
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Handle missing values
    missing_values = df_processed[numerical_cols].isnull().sum()
    if missing_values.sum() > 0:
        st.warning(f"⚠️ Missing values found. Dropping rows with missing values.")
        df_processed = df_processed.dropna(subset=numerical_cols)
        st.write(f"**Rows after handling missing values:** {len(df_processed)}")
    else:
        st.success("✅ No missing values found in numerical columns.")
    
    # Prepare features for clustering
    X = df_processed[numerical_cols].copy()
    
    # Normalize the data
    st.write("**Data Normalization:** StandardScaler is applied to normalize the features.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols, index=df_processed.index)
    
    # Display normalized data statistics
    with st.expander("📈 View Normalized Data Statistics"):
        st.dataframe(X_scaled_df.describe(), use_container_width=True)
    
    # Clustering Section
    st.markdown("---")
    st.header("🎯 K-Means Clustering")
    
    # Number of clusters input
    max_clusters = min(10, len(df_processed) // 2)  # Reasonable upper limit
    n_clusters = st.sidebar.slider(
        "Select Number of Clusters (K):",
        min_value=2,
        max_value=max_clusters,
        value=5,
        step=1,
        help="Choose the number of customer segments you want to create."
    )
    
    # Perform K-Means clustering
    if st.sidebar.button("🚀 Perform Clustering", type="primary"):
        with st.spinner("Performing K-Means clustering..."):
            # Apply K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to the dataframe
            df_clustered = df_processed.copy()
            df_clustered['Cluster'] = cluster_labels
            
            # Store in session state
            st.session_state['df_clustered'] = df_clustered
            st.session_state['kmeans'] = kmeans
            st.session_state['cluster_labels'] = cluster_labels
            st.session_state['n_clusters'] = n_clusters
    
    # Display results if clustering has been performed
    if 'df_clustered' in st.session_state:
        st.markdown("---")
        st.header("📋 Clustered Data")
        
        # Display clustered dataset
        st.subheader("Dataset with Cluster Labels")
        st.dataframe(st.session_state['df_clustered'], use_container_width=True)
        
        # Cluster statistics
        st.subheader("📊 Cluster Statistics")
        cluster_stats = st.session_state['df_clustered'].groupby('Cluster')[numerical_cols].mean()
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Cluster distribution
        st.subheader("📈 Cluster Distribution")
        cluster_counts = st.session_state['df_clustered']['Cluster'].value_counts().sort_index()
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Number of customers per cluster:**")
            for cluster, count in cluster_counts.items():
                st.write(f"Cluster {cluster}: {count} customers")
        
        with col2:
            # Bar chart for cluster distribution
            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            cluster_counts.plot(kind='bar', ax=ax_bar, color='skyblue', edgecolor='black')
            ax_bar.set_xlabel('Cluster', fontsize=12)
            ax_bar.set_ylabel('Number of Customers', fontsize=12)
            ax_bar.set_title('Customer Distribution Across Clusters', fontsize=14, fontweight='bold')
            ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=0)
            plt.tight_layout()
            st.pyplot(fig_bar)
        
        # Visualization Section
        st.markdown("---")
        st.header("📊 Cluster Visualization")
        
        # Select features for visualization
        if len(numerical_cols) >= 2:
            # Use first two numerical features for visualization
            feature1 = numerical_cols[0]
            feature2 = numerical_cols[1]
            
            st.write(f"**Visualizing clusters using:** {feature1} vs {feature2}")
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot each cluster with different color
            colors = plt.cm.tab10(np.linspace(0, 1, st.session_state['n_clusters']))
            
            for cluster_id in range(st.session_state['n_clusters']):
                cluster_data = st.session_state['df_clustered'][
                    st.session_state['df_clustered']['Cluster'] == cluster_id
                ]
                ax.scatter(
                    cluster_data[feature1],
                    cluster_data[feature2],
                    label=f'Cluster {cluster_id}',
                    s=100,
                    alpha=0.6,
                    color=colors[cluster_id],
                    edgecolors='black',
                    linewidth=0.5
                )
            
            # Plot centroids
            centroids = st.session_state['kmeans'].cluster_centers_
            # Inverse transform centroids to original scale
            centroids_original = scaler.inverse_transform(centroids)
            ax.scatter(
                centroids_original[:, numerical_cols.index(feature1)],
                centroids_original[:, numerical_cols.index(feature2)],
                marker='*',
                s=500,
                c='red',
                edgecolors='black',
                linewidth=2,
                label='Centroids'
            )
            
            ax.set_xlabel(feature1, fontsize=12, fontweight='bold')
            ax.set_ylabel(feature2, fontsize=12, fontweight='bold')
            ax.set_title('Customer Segmentation - K-Means Clustering', fontsize=16, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Additional visualization option
            if len(numerical_cols) >= 3:
                st.subheader("Alternative Visualization")
                feature1_alt = st.selectbox("Select first feature:", numerical_cols, index=0)
                feature2_alt = st.selectbox("Select second feature:", numerical_cols, index=1)
                
                if feature1_alt != feature2_alt:
                    fig2, ax2 = plt.subplots(figsize=(12, 8))
                    
                    for cluster_id in range(st.session_state['n_clusters']):
                        cluster_data = st.session_state['df_clustered'][
                            st.session_state['df_clustered']['Cluster'] == cluster_id
                        ]
                        ax2.scatter(
                            cluster_data[feature1_alt],
                            cluster_data[feature2_alt],
                            label=f'Cluster {cluster_id}',
                            s=100,
                            alpha=0.6,
                            color=colors[cluster_id],
                            edgecolors='black',
                            linewidth=0.5
                        )
                    
                    # Plot centroids
                    ax2.scatter(
                        centroids_original[:, numerical_cols.index(feature1_alt)],
                        centroids_original[:, numerical_cols.index(feature2_alt)],
                        marker='*',
                        s=500,
                        c='red',
                        edgecolors='black',
                        linewidth=2,
                        label='Centroids'
                    )
                    
                    ax2.set_xlabel(feature1_alt, fontsize=12, fontweight='bold')
                    ax2.set_ylabel(feature2_alt, fontsize=12, fontweight='bold')
                    ax2.set_title('Customer Segmentation - Custom Feature Selection', fontsize=16, fontweight='bold')
                    ax2.legend(loc='best', fontsize=10)
                    ax2.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig2)
        
        # Summary Section
        st.markdown("---")
        st.header("📝 Summary")
        st.success(f"""
        ✅ **Clustering Complete!**
        
        - **Total Customers:** {len(st.session_state['df_clustered'])}
        - **Number of Clusters:** {st.session_state['n_clusters']}
        - **Features Used:** {', '.join(numerical_cols)}
        - **Clustering Algorithm:** K-Means
        
        The customers have been successfully segmented into {st.session_state['n_clusters']} distinct groups based on their characteristics.
        """)
    
    else:
        st.info("👆 Please adjust the number of clusters using the slider in the sidebar and click 'Perform Clustering' to see the results.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>Customer Segmentation App | Built with Streamlit | K-Means Clustering</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Unable to load the dataset. Please check the file path.")

