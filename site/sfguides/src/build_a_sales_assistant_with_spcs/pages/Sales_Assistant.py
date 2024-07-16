import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import streamlit as st
from streamlit_timeline import st_timeline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# WRITE WELCOME DIRECTLY TO THE APP
title = "Sales Assistant"
page_desc = """ This is the centralized place for Sales to to explore summary of account status, including a) meeting recap b) account general status Q&A, c) competitor / contextualized search, d) account timeline
"""

st.set_page_config(
    page_title=title,
    layout='wide',
    initial_sidebar_state='expanded',
)

from tools.sales_assistant_llm_query import *
from tools.sales_assistant_prompts import *
from tools.load_data_query import *
from tools.utils import *

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.mode.copy_on_write = True # this save warning for (1).apply(determine_open_oppty_flag, axis=1) (2).sort_values(inplace)


# def sales_assistant_page():
add_colored_header(
    header=f"{title}",
    description=f"{page_desc}"
)

session = create_session_object()
time_now = datetime.now()
current_date = str(datetime.now().date())

# pull key data
df_processed_opp = fetch_data_multi_args(_session=session, query=opp_processed_query)
df_processed_opp['TOTAL_ACV_'] = df_processed_opp['TOTAL_ACV'].apply(lambda x:convert_num_to_string_flexible(x))
df_combined = fetch_data_multi_args(_session=session, query = combined_activities_query)
gong_data_competitor = fetch_data_multi_args(_session=session, query = gong_data_competitor_query)
df_combined_task = fetch_data_multi_args(_session=session, query=combined_task_query)
df_oppty_status = fetch_data_multi_args(_session=session, query = oppty_status_query)

# CREATE ACCOUNT FILTER
st.subheader(":dna: Please choose an account to start")
all_acc_name_id_dct = {}
opp_acc_name = dict(zip(df_processed_opp.ACCOUNT_NAME, df_processed_opp.ACCOUNT_ID))
combined_activity_acc_name = dict(zip(df_combined.ACCOUNT_NAME, df_combined.ACCOUNT_ID))
gong_acc_name = dict(zip(gong_data_competitor.ACCOUNT_NAME, gong_data_competitor.ACCOUNT_ID))

all_acc_name_id_dct.update(opp_acc_name)
all_acc_name_id_dct.update(combined_activity_acc_name)
all_acc_name_id_dct.update(gong_acc_name)

# Add filter: choose an account
acc_name = st.selectbox("Account Name", options = all_acc_name_id_dct.keys())
# acc_id='0010Z00001tH0VbQAK' #test aligater
# acc_id ='0010Z00001tHOk2QAG' #Flamingo
acc_id = all_acc_name_id_dct.get(acc_name)
df_processed_opp_tmp = df_processed_opp[df_processed_opp.ACCOUNT_ID==acc_id].copy()
df_combined_tmp = df_combined[df_combined.ACCOUNT_ID==acc_id].copy()
df_combined_task_tmp = df_combined_task[df_combined_task.ACCOUNT_ID == acc_id].copy()
result_gong_raw = fetch_data_multi_args(_session=session, query=preprocessed_gong_timeline_query,
                                        para_dct={"acc_id": acc_id})

context = ''
if df_processed_opp_tmp.shape[0] > 0:
    context = convert_processed_opp_to_str(df_processed_opp_tmp)
if df_combined_tmp.shape[0] > 0: # done: limit to recent 2 years tasks as they are misc.
    context = context + df_combined_tmp['ALL_ACTIVITIES_FOR_ACCT'].values[0].replace('\r\n', '')
context_cp = context

# Create tabs
tab_titles = ['Meeting Recap', 'General Question', 'Contextualized / Competitor Question', 'Account Timeline']
tab0, tab1, tab2, tab3 = st.tabs(tab_titles)


with (tab0):
    if result_gong_raw.shape[0] > 0:
        st.write('Use the checkbox on the left to see meeting details')
        data_gong_timeline = gen_grid_builder_for_gong_df(result_gong_raw)
        selected_rows = data_gong_timeline["selected_rows"]
        if len(selected_rows) != 0:
            cnt_col = 0
            col_available = []
            for col_name in ['Brief', 'Follow-up Actions']:
                if selected_rows[0][col_name]:
                    cnt_col += 1
                    col_available.append(col_name)
            cols = st.columns(cnt_col)
            record_col = 0
            for col_i in cols:
                with col_i:
                    st.markdown("##### " + col_available[record_col])
                    st.markdown(f"{selected_rows[0][col_available[record_col]]}")
                    record_col += 1
    else:
        st.write('No existing Gong conversation history to create meeting timeline.')

# GENERAL QUESTIONS
with tab1:
    st.subheader(":speech_balloon: Ask a general question about your account")

    default_general_q = ["<select a question>"]
    preselect_general_q_lst = ["Can you give me a deal summary for this account?"
                               , "What are the key risks for this deal?"
                               , "What obstacles or concerns should I know about this account?"
                               , "What was the most recent interaction we had with this client?"]

    # Setup the form for user question input and setup session_state variables
    if "text_input_general_q" not in st.session_state:
        st.session_state["text_input_general_q"] = default_general_q[0]
    if "text_select_general_q" not in st.session_state:
        st.session_state["text_select_general_q"] = default_general_q[0]

    input_preselect_general_q = default_general_q + preselect_general_q_lst
    selection_general = setup_form_universal(input_preselect=input_preselect_general_q, default_selection=default_general_q, question_name='_general_q')

    if selection_general=='' or selection_general=='No question' or  selection_general=='<select a question>':
        st.write('Please input your question to get answer.')
    if selection_general!='' and selection_general!='No question' and selection_general!='<select a question>':
        st.write("You entered question: ", selection_general)
        res_general_q_str_ = fetch_data_multi_args_general(_session=session
                              , query=mistral_query
                              , para_dct={'sys_msg':combine_context_and_quesiton_to_prompt(prompt=sys_msg_general_q, today_date=current_date)
                                        , 'user_msg': combine_context_and_quesiton_to_prompt(prompt=general_q_template
                                                                                            , question=selection_general
                                                                                            , context_i=context_cp
                                                                                            )}
                              )
        st.write(res_general_q_str_.RESULT[0].replace('$', '\$'))

# SPECIFIC CONTEXT QUESTIONS
with tab2:
    st.subheader(":hammer_and_pick: Refer below for sentiment analysis on potential competitive relationships")
    # creating competitor hashtags before allow specific questions
    idx_lst = ['SNAP_DATABRICKS',
                      'SNAP_REDSHIFT', 'SNAP_IBM',
                      'SNAP_EXASOL',
                      'SNAP_STARBURST', 'SNAP_FIREBOLT', 'SNAP_TERADATA',
                      'SNAP_ACTIAN', 'SNAP_MICROSOFT', 'SNAP_SAP',
                      'SNAP_AWS', 'SNAP_GREENPLUM', 'SNAP_GOOGLE',
                      'SNAP_CLOUDERA', 'SNAP_ORACLE', 'SNAP_VERTICA',
                      'SNAP_PALANTIR', 'SNAP_YELLOWBRICK']
    df_context_q = gong_data_competitor[(gong_data_competitor['ACCOUNT_ID'] == acc_id)].copy()
    df_context_q['COMPETITOR_CONTEXT'] = df_context_q[idx_lst].apply(lambda x: create_competitor_context(x), axis=1)

    comp_res = list(df_context_q['COMPETITOR_CONTEXT'].values)
    comp_res_set = get_unique_competitor_context_set(comp_res)
    if df_context_q.shape[0] == 0:
        st.write('There is no existing meeting details available for this account, so Sales Assistant skipped searching for competitors related context in previous conversations.')
    elif comp_res_set == {}:
        st.write('We have checked, among other data solution providers of Dog, Rabbit, Iguana, Elephant, Salamanders, Turkey, AArdvark, Moose, Starfish, Gopher, Goose, Cat, Octopus, Vulture, Penguin, Yaks none was mentioned in your previous meeting conversation.')
    else:
        st.write('We have checked if other data solution providers of Dog, Rabbit, Iguana, Elephant, Salamanders, Turkey, AArdvark, Moose, Starfish, Gopher, Goose, Cat, Octopus, Vulture, Penguin, Yaks, below are ones mentioned in your previous client meeting, click to see details:')
        availalbe_comp = list(comp_res_set.keys())
        # set the width of button to be ten
        num_of_cols=6
        if len(availalbe_comp)<num_of_cols:
            dummy_list = ['aaa_' + str(i) for i in range(num_of_cols - len(availalbe_comp))]
            availalbe_comp.extend(dummy_list)

        mask_comp_dct = {'DATABRICKS':'Vendor_Dog', 'REDSHIFT':'Vendor_Rabbit','IBM':'Vendor_Iguana','EXASOL':'Vendor_Elephant'\
                        ,'STARBURST':'Vendor_Salamanders','FIREBOLT':'Vendor_Frog','TERADATA':'Vendor_Turkey','ACTIAN':'Vendor_AArdvark'\
                        ,'MICROSOFT':'Vendor_Moose','SAP':'Vendor_Starfish','AWS':'Vendor_AArdvark','GREENPLUM':'Vendor_Gopher','GOOGLE':'Vendor_Goose'\
                        ,'CLOUDERA':'Vendor_Cat','ORACLE':'Vendor_Octopus','VERTICA':'Vendor_Vulture','PALANTIR':'Vendor_Penguin','YELLOWBRICK':'Vendor_Yaks'}
        availalbe_comp_masked = [mask_comp_dct.get(i) for i in availalbe_comp]

        pairs = zip(availalbe_comp, st.columns(len(availalbe_comp_masked)))

        for i, (comp_i, col) in enumerate(pairs):
            if comp_i.startswith('aaa_'):
                continue
            elif col.button(mask_comp_dct.get(comp_i), key=f"{comp_i}-{i}", use_container_width=True):
                # change to upload/load all of below to pre-proess
                for comp_con in comp_res_set[comp_i]: # there could be multiple places talking about the competitor
                    ###e5
                    comp_con_embedding_eb5 = fetch_data_competitor_context(_session=session, query=embedding_query, para_dct={'user_msg':comp_con[3:-3]} ).EMBEDED_RESULT.values[0]
                    df_retrieved_result = fetch_data_competitor_context(_session=session, query=similar_context_query_2,
                                                            para_dct={ 'user_msg':list(comp_con_embedding_eb5), 'acc_id':acc_id})
                    df_retrieved_result['FLAG_DOUBLE_CHECK_COMPETITOR'] = df_retrieved_result['MEETING_CHUNK'].apply(lambda x: double_check_competitor(x, comp_i))
                    df_retrieved_res_checked = df_retrieved_result[df_retrieved_result['FLAG_DOUBLE_CHECK_COMPETITOR']==True]

                    # Done: THIS IF CONDITION SHOULD BE REMOVED, AS THERE HAVE TO BE BUZZ WORD FOUND, NEED TO CHANGE CONTAINS() TO SEARCH FOR a dct value
                    if df_retrieved_res_checked.shape[0]>0:
                        context_competitor_checked = convert_multiple_pieces_to_one_feed(df_retrieved_res_checked)
                        # st.write('* Sales Assistant found '+ comp_i +' was mentioned as below: ')
                        if isinstance(context_competitor_checked, str):
                            # st.write(context_competitor_checked)
                            st.write("* Sales Assistant's comprehension on " + mask_comp_dct.get(comp_i) + ":")
                            result_context_competitor_q_str = fetch_data_competitor_context(_session = session
                                                                   , query=mistral_query
                                                                   , para_dct={'sys_msg':sys_msg_retrival
                                                                             , 'user_msg':combine_context_and_quesiton_to_prompt(prompt=user_retrival_specific_competitor_template
                                                                                                                            , question=comp_i
                                                                                                                            , context_i=context_competitor_checked)}
                                                                                    ).RESULT.values[0]

                        else:
                            st.write("* Sales Assistant's comprehension on " + mask_comp_dct.get(comp_i) + ":")
                            result_context_competitor_q_str = fetch_data_competitor_context(_session = session
                                                                   , query=mistral_query
                                                                   , para_dct={'sys_msg':sys_msg_retrival
                                                                           , 'user_msg':combine_context_and_quesiton_to_prompt(prompt=user_retrival_specific_competitor_template
                                                                                        , question=comp_i
                                                                                        , context_i=context_competitor_checked.values)}
                                                                   ).RESULT.values[0]
                        #maskout competitor before show
                        mask_comp_reverse_dct = {'Vendor_Dog': ['DATABRICKS','Databricks','Databrick','data brick',]
                                                 ,'Vendor_Rabbit': ['REDSHIFT','Redshift','redshift']
                                                 ,'Vendor_Iguana': ['IBM',' ibm ']
                                                 ,'Vendor_Elephant': ['EXASOL','Exasol','exasol']
                                                 ,'Vendor_Salamanders': ['STARBURST','Starburst']
                                                 ,'Vendor_Frog': ['FIREBOLT','Firebolt','firebolt']
                                                 ,'Vendor_Turkey': ['TERADATA','Teradata']
                                                 ,'Vendor_AArdvark': ['AWS',' aws ','aws emr','AWS EMR','ACTIAN','Actian','actian']
                                                 ,'Vendor_Moose': ['MICROSOFT','Microsoft','microsoft','micro soft','Azure','azure']
                                                 ,'Vendor_Starfish': ['SAP',' sap ']
                                                 ,'Vendor_Gopher': ['GREENPLUM', 'greenplum']
                                                 ,'Vendor_Goose': ['GOOGLE','Google']
                                                 ,'Vendor_Cat': ['CLOUDERA','Cloudera','cloudera']
                                                 ,'Vendor_Octopus': ['ORACLE','Oracle','oracle']
                                                 ,'Vendor_Vulture': ['VERTICA', 'Vertica ',' vertica ']
                                                 ,'Vendor_Penguin': ['PALANTIR','Palantir','palantir']
                                                 ,'Vendor_Yaks': ['YELLOWBRICK','Yellowbrick',' yellow brick ',' yellowbrick ']}
                        for real_comp_i in mask_comp_reverse_dct[mask_comp_dct.get(comp_i)]:
                            result_context_competitor_q_str = result_context_competitor_q_str.replace(real_comp_i, mask_comp_dct.get(comp_i))
                        st.write(result_context_competitor_q_str)

    st.subheader(":hammer_and_pick: Ask / search more detailed context about your account ")
    if df_context_q.shape[0]==0:
        st.write('There is no existing meeting details available for this account, so Sales Assistant will not able to provide use sentiment Q&A in below box.')

    default_context_q = ["<select a question>"]
    preselect_context_q_lst = ["Why is Snowpark Container Service mentioned?",
                               "What does the client want to do about integration and data sharing?",
                               "What are the product features from CompetitorABC the prospect client is thinking to use?",
                               "Has the client show any interest or plan on GenAI or large language model?",
                               "I remember the client mentioned a conference, can you remind me of the context?"]

    # Setup the form for user question input and setup session_state variables
    if "text_input_context_q" not in st.session_state:
        st.session_state["text_input_context_q"] = default_context_q[0]
    if "text_select_context_q" not in st.session_state:
        st.session_state["text_select_context_q"] = default_context_q[0]

    input_preselect_context_q = default_context_q + preselect_context_q_lst
    context_question = setup_form_universal(input_preselect=input_preselect_context_q,
                                            default_selection=default_context_q, question_name='_context_q')

    if context_question=='' or context_question=='No question' or  context_question=='<select a question>':
        st.write('Please input your question to get answer.')
    else:
        st.write("You entered question: ", context_question)
        # query_question_embedding = create_embedding_to_query(context_question)
        query_question_embedding = fetch_data_multi_args_context(_session=session, query=embedding_query, para_dct={'user_msg':context_question}).EMBEDED_RESULT.values[0]
        df_retrieved_result = fetch_data_multi_args_context(_session=session, query=similar_context_query_2
                                                , para_dct={'user_msg':list(query_question_embedding),'acc_id':acc_id})
        # query after get the question
        context_combined = convert_multiple_pieces_to_one_feed(df_retrieved_result.head(3))
        result_context_q_str = fetch_data_multi_args_context(_session = session
                                                    , query=mistral_query
                                                    , para_dct={'sys_msg':sys_msg_retrival
                                                    , 'user_msg':combine_context_and_quesiton_to_prompt(
                                                                    prompt=user_retrival_specific_template
                                                                            , question=context_question
                                                                            , context_i=context_combined)}
                                                    ).RESULT.values[0]
        st.write(result_context_q_str)


# TIMELINE:
with tab3:
    window_dct = {'All':365*10, 'Last 1 Year':365, 'Last 6 Months':365*0.5, 'Last 3 Months': 91}
    timeline_window = st.radio("Select the activity window",window_dct.keys(), index=1)
    cutoff_date = str(date.today() - timedelta(days=window_dct.get(timeline_window)))

    subtab_titles = ['Task Timeline', 'Opportunity Next Step Timeline']
    # tab_task_timeline, tab_opp_nextstep_timeline, tab_gong_timeline = st.tabs(subtab_titles)
    tab_task_timeline, tab_opp_nextstep_timeline = st.tabs(subtab_titles)

    with tab_task_timeline:
        if df_combined_task_tmp.shape[0] > 0:
            result_task_df = df_combined_task_tmp[['CLEANED_TASK','SUBJECT_','ACTIVITY_DATE_']]\
                                                .rename(columns={'ACTIVITY_DATE_':'start'
                                                                  , 'SUBJECT_': 'content'
                                                                  , 'CLEANED_TASK':'task details'})

            timeline_task = st_timeline(result_task_df[result_task_df.start>=cutoff_date].to_dict('records'), groups=[], options={}, height="400px")
            st.write("To see details, select a tag in timeline by clicking on the label")
            st.write(timeline_task)

        else:
            st.write('No qualified task data to create task timeline.')

    with tab_opp_nextstep_timeline:
        result_opp_raw = fetch_data_multi_args(_session = session, query=preprocessed_nextsteps_timeline_query, para_dct= {'acc_id': acc_id})

        if result_opp_raw.shape[0] > 0:
            result_opp_df = rename_opp_timeline(result_opp_raw)
            timeline_opp = st_timeline(result_opp_df[result_opp_df.start>=cutoff_date].to_dict('records'), groups=[], options={}, height="400px")
            st.write('The label of each next step is summarized using LLM based on user input of next steps in opportunity field.')
            st.write("To see details, select a tag in timeline by clicking on the label")
            st.write(timeline_opp)

        else:
            st.write('No existing Next Steps data to create opportunity timeline.')