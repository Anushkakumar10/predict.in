import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import io
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title='predict.in',
                   layout='wide',
                   initial_sidebar_state='expanded',
                   menu_items={
                       'Report a bug': "mailto:arnav.vatsal2213@gmail.com",
                       'About': '''# Welcome to predict.in
Create predictive models using AI. Upload a dataset and leave the rest to us. You are in good hands.
\nDeveloped with ❤️ by **Arnav**.
\nhttps://github.com/arnav003
'''
                   }
                   )

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


@st.cache_resource
def eda(df, explorative, mininmal):
    pr = ProfileReport(df, explorative=explorative, minimal=mininmal)
    return pr


def build_model(X, Y, split_size, seed_number):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=seed_number)
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models_train, predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test, predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)
    return predictions_train, predictions_test


def plot_model_details(predictions_train, predictions_test):
    st.subheader('Table of Model Performance')

    st.write('Training set')
    st.write(predictions_train)
    st.markdown(filedownload(predictions_train, 'training.csv'), unsafe_allow_html=True)

    st.write('Test set')
    st.write(predictions_test)
    st.markdown(filedownload(predictions_test, 'test.csv'), unsafe_allow_html=True)

    st.subheader('3. Plot of Model Performance (Test set)')

    with st.markdown('**R-squared**'):
        # Tall
        predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"]]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
        ax1.set(xlim=(0, 1))
    st.markdown(imagedownload(plt, 'plot-r2-tall.pdf'), unsafe_allow_html=True)
    # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
    ax1.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt, 'plot-r2-wide.pdf'), unsafe_allow_html=True)

    with st.markdown('**RMSE (capped at 50)**'):
        # Tall
        predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"]]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
    st.markdown(imagedownload(plt, 'plot-rmse-tall.pdf'), unsafe_allow_html=True)
    # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt, 'plot-rmse-wide.pdf'), unsafe_allow_html=True)

    with st.markdown('**Calculation time**'):
        # Tall
        predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"]]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
    st.markdown(imagedownload(plt, 'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
    # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt, 'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)


def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
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

# with st.expander('Show session state variables'):
#     st.write(f"{st.session_state}")

st.write('''
# predict.in  
Get to know your dataset!
''')


# Upload CSV data
def on_uploaded_file_change():
    st.session_state['df_uploaded'] = 'no'
    st.session_state['df_columns'] = []
    st.session_state['required_df_columns'] = []
    st.session_state['df_index'] = None
    st.session_state['df_target'] = None
    st.session_state['columns_to_remove'] = []
    st.session_state['columns_to_add_back'] = []


with st.sidebar.header('Select data'):
    if st.sidebar.checkbox("Use example dataset"):
        uploaded_file = st.sidebar.selectbox('Select dataset:',
                                             options=['Dataset/titanic.csv',
                                                      'Dataset/home_data.csv'
                                                      ],
                                             on_change=on_uploaded_file_change
                                             )
    else:
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
                st.session_state['required_df_columns'] = remove_columns(st.session_state['columns_to_remove']).copy()
                df = df.set_index(index)
                if st.session_state['df_index'] is not None:
                    st.session_state['required_df_columns'] = add_back_columns([index]).copy()
                st.session_state['df_index'] = index

        target = st.sidebar.selectbox('Select target column',
                                      [None, ] + st.session_state['required_df_columns'])
        if target is None:
            if st.session_state['df_target'] is not None:
                target = st.session_state['df_target']
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

    st.header('**Input DataFrame**')

    if target is None:
        st.warning("Target variable not defined. Set Target column.")
        st.write(df)
    elif target == index:
        st.warning("Index and Target columns can't be same.")
        st.write(df)
    else:
        X, Y = create_x_y(df, target)

        col1, col2 = st.columns([4, 1])

        col1.write('X')
        col1.write(X)
        col1.write(X.shape)

        col2.write('Y')
        col2.write(Y)
        col2.write(Y.shape)

        st.write('Variable details')
        st.write('X variable')
        st.info(list(X.columns))
        st.write('Y variable')
        st.info(Y.name)

    st.write('---')

    mininmal_eda = st.checkbox('Minimal EDA')
    if st.checkbox('Genrate EDA'):
        pr = eda(df, False, mininmal_eda)
        st.header('**Profiling Report**')
        st_profile_report(pr)

        desc = pr.get_description()

        with st.expander('''Table details'''):
            table_details = desc.table
            print(table_details)

            no_of_rows = table_details['n']
            no_of_columns = table_details['n_var']
            no_of_cells = no_of_rows * no_of_columns
            no_of_missing_cells = table_details['n_cells_missing']
            no_of_missing_cells_percentage = np.round((no_of_missing_cells * 100) / (no_of_columns * no_of_rows), 2)
            no_of_columns_with_missing_data = table_details['n_vars_with_missing']
            types_of_columns = table_details['types']
            no_of_duplicate_rows = table_details['n_duplicates']
            no_of_duplicate_rows_percentage = table_details['p_duplicates']

            st.write(f'''
            Number of observations / rows: {no_of_rows} \n
            Number of variables / columns: {no_of_columns} \n
            Number of cells: {no_of_cells} \n
            Number of missing cells: {no_of_missing_cells}  ({no_of_missing_cells_percentage} %) \n            
            Number of columns with missing data: {no_of_columns_with_missing_data} \n                 
            Number of duplicate rows: {no_of_duplicate_rows}  ({no_of_duplicate_rows_percentage} %) \n  
            ''')

            # for key in table_details:
            #     st.write(f'''{key} : {table_details[key]}''')

        with st.expander('''Alerts'''):
            alerts_list = desc.alerts
            for i, item in enumerate(alerts_list):
                if item.alert_type_name in {'Missing', 'Zeros'}:
                    st.markdown(f"- {item}")
                    col1, col2 = st.columns([1, 3])
                    if col1.checkbox(f'Fix {item.column_name} column'):
                        if col2.button('Delete entire column', key=f'delete entire {item.column_name} button'):
                            col2.success(f'Deleted {item.column_name}')
                        if col2.button('Delete rows where values are missing', key=f'delete {item.column_name} button'):
                            col2.success(f'Deleted rows with missing values in {item.column_name} column')
                        if col2.button('Fill missing values', key=f'fill {item.column_name} button'):
                            value = None
                            if col2.button('Fill with mean value', key=f'fill mean {item.column_name} button'):
                                value = 12
                            if col2.button('Fill with median value', key=f'fill median {item.column_name} button'):
                                value = 9
                            if value is not None:
                                col2.success(f'Filled missing values in {item.column_name} with {value}')
                    else:
                        st.warning('Column with missing value found.')

    st.write('---')

    if st.checkbox('Build models'):
        predictions_train, predictions_test = build_model(X, Y, split_size, seed_number)
        plot_model_details(predictions_train, predictions_test)
