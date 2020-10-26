import base64
import numpy as np
import os
import pandas as pd
import streamlit as st
from string_grouper import match_strings, group_similar_strings


def main():
    # Page configuration
    st.beta_set_page_config(page_title='String Search', page_icon='🔍', layout='centered', initial_sidebar_state='expanded')

    # Title
    st.title('String Search')
    st.markdown("""
    ### An app that lets you search and clean up your datasets
    """)
    st.write("---")

    # Sidebar
    st.sidebar.title('User Input')
    st.sidebar.subheader('Source File')
    file_option = st.sidebar.radio('Default Files or Upload Files?', ['Default', 'Upload'])

    st.sidebar.subheader('Advanced Options')
    duplicated_option = st.sidebar.checkbox('Check for Duplicates')
    group_option = st.sidebar.checkbox('Check for Group Matches (Inclusive Duplicates)')

    st.sidebar.title('About')
    st.sidebar.markdown("""
    * **Python libraries:** `base64`, `numpy`, `pandas`, `string_grouper` and `streamlit`
    * **Data source:** [7+ Million Company Dataset](https://www.kaggle.com/peopledatalabssf/free-7-million-company-dataset).
    * **Reference:** [Super Fast String Matching in Python](https://bergvca.github.io/2017/10/14/super-fast-string-matching.html).
    """)

    if file_option == 'Default':
        folder_path = './data'
        filenames = sorted(os.listdir(folder_path))
        selected_filename = st.selectbox('Select CSV file', filenames, filenames.index('companies_sorted_my.csv'))
        file = os.path.join(folder_path, selected_filename)
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type='csv')
        file = None
        if uploaded_file is not None:
            uploaded_file.seek(0)
            file = uploaded_file

    if file is not None:
        df = load_data(file)
        st.info(f'Data Dimension: {df.shape[0]:,d} rows and {df.shape[1]:,d} columns.')

        # Create copy df and insert row number
        dfn = df.copy()
        dfn.insert(loc=0, column='row_num', value=np.arange(len(dfn))+1)

        # Search section
        search_expander = st.beta_expander(label='Search', expanded=True)
        with search_expander:
            c1, c2, c3 = st.beta_columns((1, 3, 1))

            columns = df.columns.tolist()
            selected_column = c1.selectbox('Select column', columns)
            options = ['Starts with', 'Contains', 'Most similar']
            selected_option = c3.selectbox('Options', options)

            # User input
            user_input = c2.text_input('Search here')
            if user_input:
                df_selected = get_matches(dfn, selected_column, user_input, selected_option)
            else:
                df_selected = df.copy()

            st.write(df_selected)
            write_footer(df_selected)

        # Duplicates section
        if duplicated_option:
            duplicated_expander = st.beta_expander(label='Duplicates', expanded=True)
            with duplicated_expander:
                st.info("""
                This section retrieve the duplicate records based on selected column(s). 
                Single or multiple columns can be used for duplication check.
                """)

                # Column selection
                selected_column_duplicated = st.multiselect('Select column(s)', columns, selected_column)

                # Check duplicated rows
                if selected_column_duplicated:
                    duplicated_df = get_duplicated(dfn, selected_column_duplicated)
                    st.write(duplicated_df)
                    write_footer(duplicated_df)

                    other_duplicated = st.checkbox('View remaining records i.e. exclude the above records')
                    if other_duplicated:
                        other_duplicated_df = get_others(dfn, duplicated_df[dfn.columns])
                        st.write(other_duplicated_df)
                        write_footer(other_duplicated_df)

        # Group matches section
        if group_option:
            group_expander = st.beta_expander(label='Group Matches', expanded=True)
            with group_expander:
                st.info("""
                This section group each records based on selected column if they are found 
                to be closely similar or duplicates.
                """)

                selected_column_group = st.selectbox('Select column', columns, index=columns.index(selected_column), key='group')

                group_df = get_group(dfn, selected_column_group)
                st.write(group_df)
                write_footer(group_df)

                other_group = st.checkbox('View remaining records i.e. exclude the above records', key='group')
                if other_group:
                    other_group_df = get_others(dfn, group_df[dfn.columns])
                    st.write(other_group_df)
                    write_footer(other_group_df)
    
    st.sidebar.write("---")
    st.sidebar.info(""" 
    by: [Rizuwan Zulkifli](https://www.linkedin.com/in/rizuwanzul/) | source: [GitHub](https://github.com/rizuwanzul/string-search)
    """)


@st.cache
def load_data(file):
    df = pd.read_csv(file)
    format = lambda x: str(x).lower().replace(' ', '_')
    df.rename(format, axis='columns', inplace=True)

    return df


def get_matches(df, col, input, option):

    if option == 'Starts with':
        cond = df[col].astype(str).str.lower().str.startswith(input.lower())
        dfn = df[cond].copy()
        dfn.sort_values(by=col, inplace=True, ignore_index=True)
        
    elif option == 'Contains':
        cond = df[col].astype(str).str.lower().str.contains(input.lower())
        dfn = df[cond].copy()
        dfn.sort_values(by=col, inplace=True, ignore_index=True)

    elif option == 'Most similar':
        dfn = match_strings(df[col].astype(str).drop_duplicates(), pd.Series(input), min_similarity=0.4)
        cols = dfn.columns
        dfn.sort_values(by='similarity', ascending=False, inplace=True, ignore_index=True)
        dfn = pd.merge(dfn, df, left_on='left_side', right_on=col)
        dfn.drop(cols, axis=1, inplace=True)

    return dfn


@st.cache
def get_duplicated(df, cols):
    dfn = df[df.duplicated(subset=cols, keep=False)].sort_values(by=cols+['row_num'], ignore_index=True)
    sets = dfn[cols].apply(tuple, axis=1).rank(method='dense').astype(int)
    dfn.insert(loc=0, column='set', value=sets)

    return dfn


@st.cache
def get_group(df, col):
    dfn = df.copy()
    dfn['deduplicated'] = group_similar_strings(dfn[col].astype(str))

    dfn = dfn.groupby('deduplicated').filter(lambda x: len(x) > 1).sort_values(by=['deduplicated', 'row_num'], ignore_index=True)
    sets = dfn['deduplicated'].rank(method='dense').astype(int)
    dfn.insert(loc=0, column='set', value=sets)
    dfn.drop(['deduplicated'], axis=1, inplace=True)

    return dfn


# Get difference between two DFs
@st.cache
def get_others(df1, df2):
    return pd.concat([df1, df2, df2]).drop_duplicates(keep=False)


# Export data
def download_file(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Export data as CSV file</a>'
    return href


def write_footer(df):
    c1, c2 = st.beta_columns((1, 1))
    c1.write(f'Total matches: {df.shape[0]:,d} records')
    c2.markdown(f"<p style='text-align: right;'>{download_file(df)}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
