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
if 'df_uploaded' not in st.session_state:
    st.session_state['df_uploaded'] = 'no'  ## ['yes', 'changed', 'no']
if 'df_columns' not in st.session_state:
    st.session_state['df_columns'] = []
if 'required_df_columns' not in st.session_state:
    st.session_state['required_df_columns'] = []
if 'df_index' not in st.session_state:
    st.session_state['df_index'] = None
if 'df_target' not in st.session_state:
    st.session_state['df_target'] = None
if 'columns_to_remove' not in st.session_state:
    st.session_state['columns_to_remove'] = []
if 'columns_to_add_back' not in st.session_state:
    st.session_state['columns_to_add_back'] = []


def create_x_y(df, target):
    columns = st.session_state['required_df_columns'].copy()
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


## Returns the df columns after removing the columns in columns_to_remove
def remove_columns(columns_to_remove):
    columns = st.session_state['df_columns'].copy()
    for col in columns_to_remove:
        columns.remove(col)
    return columns


## Returns the df columns after adding back the columns from columns_to_add_back
def add_back_columns(columns_to_add_back):
    updated_columns_to_remove = [x for x in st.session_state['columns_to_remove'] if x not in columns_to_add_back]
    st.session_state['columns_to_remove'] = updated_columns_to_remove.copy()
    columns = remove_columns(updated_columns_to_remove).copy()
    st.session_state['columns_to_add_back'] = []
    return columns


## Printing session states
print('1', st.session_state['df_uploaded'])
print('2', st.session_state['df_columns'])
print('3', st.session_state['required_df_columns'])
print('4', st.session_state['df_index'])
print('5', st.session_state['df_target'])
print('6', st.session_state['columns_to_remove'])
print('7', st.session_state['columns_to_add_back'])

st.write('''
# predict.in

Get to know your dataset!
''')


# Upload CSV data
def on_uploaded_file_change():
    st.session_state['df_uploaded'] = 'no'


with st.sidebar.header('Upload your data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file",
                                             type=["csv"],
                                             on_change=on_uploaded_file_change)

if uploaded_file is None:
    st.info('Awaiting for CSV file to be uploaded.')
else:
    def load_csv():
        df = pd.read_csv(uploaded_file)
        if st.session_state['df_uploaded'] == 'no':
            st.session_state['df_columns'] = list(df.columns).copy()
            st.session_state['required_df_columns'] = list(df.columns).copy()
            st.session_state['df_uploaded'] = 'yes'
        return df


    df = load_csv()

    with st.sidebar.header('Set Parameters'):
        index = st.sidebar.selectbox('Select index column',
                                     [None, ] + st.session_state['df_columns'])
        if index is not None:
            if index in st.session_state['required_df_columns']:
                st.session_state['columns_to_remove'].extend([index])
                st.session_state['required_df_columns'] = remove_columns([index]).copy()
                df = df.set_index(index)
                if st.session_state['df_index'] is not None:
                    st.session_state['required_df_columns'] = add_back_columns([index]).copy()
                st.session_state['df_index'] = index



        target = st.sidebar.selectbox('Select target column',
                                      [None, ] + st.session_state['required_df_columns'])
        if target is None:
            if st.session_state['df_target'] is not None:
                target=st.session_state['df_target']
        elif target is not None:
            st.session_state['df_target'] = target

        columns_to_remove = st.sidebar.multiselect('Select columns to remove',
                                                   st.session_state['required_df_columns'])

        if st.sidebar.button('Remove selected columns'):
            st.session_state['columns_to_remove'].extend(columns_to_remove)
            st.session_state['required_df_columns'] = remove_columns(st.session_state['columns_to_remove']).copy()


        if len(st.session_state['columns_to_remove']) != 0:
            columns_to_add_back = st.sidebar.multiselect('Select columns to add back',
                                                         st.session_state['columns_to_remove'])

            if st.sidebar.button('Add back selected columns'):
                st.session_state['columns_to_add_back'].extend(columns_to_add_back)
                st.session_state['required_df_columns'] = add_back_columns(columns_to_add_back).copy()

        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)
    ## df after setting parameters
    df = df.loc[:, st.session_state['required_df_columns']]
    try:
        st.header('**Input DataFrame**')
        if target != index:
            if target is not None:
                X, Y = create_x_y(df, target)
        else:
            st.warning("Index and Target columns can't be same.")
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

    st.markdown(''' # Work in Progress
    
        if st.checkbox('Build models'):
        
        predictions_train, predictions_test = build_model(X, Y, split_size, seed_number)

        st.subheader('Table of Model Performance')

        st.write('Training set')
        st.write(predictions_train)
        st.markdown(filedownload(predictions_train, 'training.csv'), unsafe_allow_html=True)

        st.write('Test set')
        st.write(predictions_test)
        st.markdown(filedownload(predictions_test, 'test.csv'), unsafe_allow_html=True)
    ''')
