import streamlit as st
import pandas as pd
from quickclus import QuickClus
import numpy as np



@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


st.title("Auto clustering example")
example_file = pd.read_csv("1000 Sales Records.csv")
example_columns = example_file.columns
example_file = convert_df(example_file)

st.download_button(
    label="Example file",
    data=example_file,
    file_name='1000 Sales Records.csv',
    mime='text/csv',
)

uploaded_file = st.file_uploader("Choose a file")

c1, c2 = st.columns(2)
with c1:
    optimize_model = st.selectbox("Optimize model", ["Yes", "No"], index=1, help="Select whether the model should use the default values (No) or try to find parameters that generate more differentiated clusters (Yes).")
    optimize_model = optimize_model == "Yes"
with c2:
    describe_clusters = st.selectbox("Describe clusters", ["Yes", "No"], index=0, help="Indicate if you want to generate automatic descriptions of the different clusters found. If the number of clusters is too large, it may be difficult to read.")
    describe_clusters = describe_clusters == "Yes"

if uploaded_file is not None:
    if "csv" in uploaded_file.name:
        df = pd.read_csv(uploaded_file)
    elif "xls" in uploaded_file.name:
        df = pd.read_excel(uploaded_file)
    else:
        st.write("Only csv or xls files are supported. Upload a file with one of these extensions to continue.")
        st.stop()

    with st.expander("Show uploaded data"):
        st.dataframe(df.head())

    cols_to_drop_help = "Select the columns to exclude from the analysis. Excluding columns with unique values (such as ids) or columns with dates/times is recommended."

    if (example_columns == df.columns).all():
        cols_to_drop = st.multiselect(
        'Columns to drop',
        df.columns,
        ["Order Date", "Order ID", "Ship Date"],
        help = cols_to_drop_help)
    else:
        cols_to_drop = st.multiselect(
        'Columns to drop',
        df.columns,
        [],
        help = cols_to_drop_help)

    with st.expander("Advanced options"):
        st.write("UMAP parameters")
        c1, c2 = st.columns(2)
        with c1:
            n_components = st.number_input("N° components", min_value = 1, value = 2)
            n_neighbors = st.number_input("N° neighbors", min_value = 0, value = 15)
            scaler_type_numerical = st.selectbox("Scaler numerical",
                                 options = ["standard", "robust"], index = 0)
            transformation_type_numerical = st.selectbox("Transformation numerical",
                                 options = ["power", "quantile"], index = 0)
            
        with c2:
            
            threshold_combine_rare_levels = st.number_input("Threshold categorical level", 
                                min_value = 0.0, max_value = 1.0, value = 0.0)
            imputer_strategy_numerical = st.selectbox("Imputation numerical",
                                 options = ["mean", "median", "most_frequent"], index = 0)
            umap_combine_method = st.selectbox("UMAP combine method",
                                 options = ["intersection", "union", "contrast", "intersection_union_mapper"], index = 0)
            n_neighbors_intersection_union = st.number_input("N° neighbors intersection", min_value = 0, value = 15)

        st.write("\n")
        st.write("HDBSCAN parameters")
        c1, c2 = st.columns(2)
        with c1:
            min_cluster_size = st.number_input("Min cluster size", min_value = 0, value = 15)
            if optimize_model:
                n_trials = st.number_input("N° trials optimization", min_value = 0,
                             max_value = 300, value = 100)

        with c2:
            min_samples = st.number_input("Min samples", min_value = 0, value = 15)
    
    if st.button('Run analysis'):
        #clf = QuickClus(n_components = 2)
        clf = QuickClus(n_neighbors = n_neighbors, min_cluster_size = min_cluster_size,
                min_samples = min_samples,
                threshold_combine_rare_levels = threshold_combine_rare_levels,
                n_components = n_components,
                scaler_type_numerical = scaler_type_numerical,
                imputer_strategy_numerical = imputer_strategy_numerical,
                transformation_type_numerical = transformation_type_numerical,
                umap_combine_method = umap_combine_method,
                n_neighbors_intersection_union = None,
                verbose = False)

        if len(cols_to_drop) > 0:
            clf.fit(df.drop(cols_to_drop, axis = 1))
        results = clf.assing_results(df)

        if optimize_model:
            st.subheader("Before optimization")

        st.pyplot(clf.plot_2d_labels())

        if optimize_model:
            clf.tune_model(n_trials=n_trials)
            st.subheader("After optimization")
            results = clf.assing_results(df)
            st.pyplot(clf.plot_2d_labels())


        numerics_cols = results.select_dtypes(include = [int, float]).drop(["Cluster"], 1).columns.tolist()

        cat_cols = results.select_dtypes(exclude = ["float", "int", "datetime"]).columns.tolist()
            
        columns_analyze_numerical = [c for c in numerics_cols if c not in cols_to_drop]

        columns_analyze_categorical = [c for c in cat_cols if c not in cols_to_drop]

        with st.expander("Show descriptions"):
            if describe_clusters:
                results_describe = clf.describe_cluster(results_df = results, clusters = np.sort(results["Cluster"].unique()),
                                        columns_analyze_numerical = columns_analyze_numerical,
                                        columns_analyze_categorical = columns_analyze_categorical, 
                                        metric = "mean")

                for element in results_describe.keys():
                    st.markdown(f"**Cluster {element}**")
                    st.markdown(results_describe[element].replace(".", ".\n").replace(":", ":\n"))
                    st.write("\n")

        csv_output = convert_df(results)

        st.subheader("Data preview")
        st.dataframe(results.head())
        st.download_button(
            label="Download results as csv",
            data=csv_output,
            file_name='results.csv',
            mime='text/csv',
        )
        
st.markdown("""---""")
st.write("Do you have feedback? Please send me a message on [LinkedIn](https://www.linkedin.com/in/aitorporcellaburu/).")


