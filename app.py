import streamlit as st
import pandas as pd
import numpy as np
import base64
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title='predict.in', layout='wide', initial_sidebar_state='expanded')

#### Session States ####
if 'columns_to_remove' not in st.session_state:
    st.session_state['columns_to_remove'] = []

def create_x_y(df, target):
    columns = list(df.columns)
    x_columns = columns
    x_columns.remove(target)
    X = df.loc[:, x_columns]  # Using all column except target column as X
    Y = df.loc[:, target]  # Selecting the target column as Y
    return X, Y

def eda(df):
    pr = ProfileReport(df, explorative=True)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)

def build_model(X, Y, split_size, seed_number):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=seed_number)
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models_train, predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test, predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)
    return predictions_train, predictions_test

def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def remove_columns(df, columns_to_remove):
    columns=list(df.columns)
    for col in columns_to_remove:
        columns.remove(col)
    df = df.loc[:, columns]
    return df

st.write('''
# predict.in

Get to know your dataset!
''')

# Upload CSV data
with st.sidebar.header('Upload your data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    def load_csv():
        df = pd.read_csv(uploaded_file)
        df = remove_columns(df, st.session_state['columns_to_remove'])
        return df


    df = load_csv()
    df_columns = [None, ] + list(df.columns)

    with st.sidebar.header('Set Parameters'):
        index = st.sidebar.selectbox('Select index column', df_columns)
        if index is not None:
            df = df.set_index(index)

        target = st.sidebar.selectbox('Select target column', df_columns)
        if target is not None:
            X, Y = create_x_y(df, target)

        #split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        #seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)

        def on_columns_to_remove_change():
            st.session_state['columns_to_remove'] = columns_to_remove
        columns_to_remove = st.sidebar.multiselect('Select columns to remove',
                                                   df.columns,
                                                   on_change=on_columns_to_remove_change)

        if st.sidebar.button('Remove selected columns'):
            df = remove_columns(df, columns_to_remove)

    try:
        st.header('**Input DataFrame**')
        if target is None:
            st.write(df)
        else:
            col1, col2 = st.columns([4, 1])
            col1.write('X')
            col1.write(X)
            col1.write(X.shape)
            col2.write('Y')
            col2.write(Y)
            col2.write(Y.shape)
    except NameError:
        st.warning('Target variable not defined.')


    if st.checkbox('Genrate EDA'):
        eda(df)


    print('''if st.checkbox('Build models'):
        predictions_train, predictions_test = build_model(X, Y, split_size, seed_number)

        st.subheader('Table of Model Performance')

        st.write('Training set')
        st.write(predictions_train)
        st.markdown(filedownload(predictions_train, 'training.csv'), unsafe_allow_html=True)

        st.write('Test set')
        st.write(predictions_test)
        st.markdown(filedownload(predictions_test, 'test.csv'), unsafe_allow_html=True)''')

else:
    st.info('Awaiting for CSV file to be uploaded.')
