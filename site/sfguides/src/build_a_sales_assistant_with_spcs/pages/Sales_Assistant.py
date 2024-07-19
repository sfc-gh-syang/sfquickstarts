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

# todo: remove after lock
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.mode.copy_on_write = True


add_colored_header(
    header=f"{title}",
    description=f"{page_desc}"
)

session = create_session_object()
time_now = datetime.now()
current_date = str(datetime.now().date())

# pull key data
conv_data_competitor = fetch_data_multi_args(_session=session, query = conv_data_competitor_query)

# CREATE ACCOUNT FILTER
st.subheader(":dna: Please choose an account to start")
all_acc_name_id_dct = {}
conv_acc_name = dict(zip(conv_data_competitor.ACCOUNT_NAME, conv_data_competitor.ACCOUNT_ID))
all_acc_name_id_dct.update(conv_acc_name)


# Add filter: choose an account
acc_name = st.selectbox("Account Name", options = all_acc_name_id_dct.keys())
acc_id = all_acc_name_id_dct.get(acc_name)

# Create tabs
tab_titles = ['Meeting Recap', 'General Question', 'Contextualized / Competitor Question', 'Account Timeline']
tab0, tab1, tab2, tab3 = st.tabs(tab_titles)


# MEETING LIST
with (tab0):
    result_conv_raw = fetch_data_multi_args(_session=session, query=preprocessed_conv_timeline_query,
                                            para_dct={"acc_id": acc_id})
    if result_conv_raw.shape[0] > 0:
        st.write('Use the checkbox on the left to see meeting details')
        data_conv_timeline = gen_grid_builder_for_conv_df(result_conv_raw)
        selected_rows = data_conv_timeline["selected_rows"]
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
        st.write('No existing conversation history to create meeting timeline.')

# GENERAL QUESTIONS
with tab1:
    st.subheader(":speech_balloon: Ask a general question about your account")
    # prepare context_i
    df_combined_tmp = fetch_data_multi_args(_session=session
                                            , query=combined_activities_query,
                                            para_dct={'acc_id': acc_id})
    df_processed_opp_tmp = fetch_data_multi_args(_session=session
                                                 , query=opp_processed_query
                                                 , para_dct= {'acc_id': acc_id})
    context_i = ''
    if df_processed_opp_tmp.shape[0] > 0:
        context_i = convert_processed_opp_to_str(df_processed_opp_tmp)
    if df_combined_tmp.shape[0] > 0:
        context_i = context_i + df_combined_tmp['ALL_ACTIVITIES_FOR_ACCT'].values[0].replace('\r\n', '')


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
                                                                                            , context_i=context_i
                                                                                            )}
                              )
        st.write(res_general_q_str_.RESULT[0].replace('$', '\$'))


# SPECIFIC CONTEXT QUESTIONS
with tab2:
    st.subheader(":hammer_and_pick: Refer below for sentiment analysis on potential competitive relationships")
    # creating competitor hashtags before allow specific questions
    idx_lst = ['SNAP_DA20', 'SNAP_RE4', 'SNAP_I2', 'SNAP_EX1',
               'SNAP_ST1', 'SNAP_FI18', 'SNAP_TE18',
               'SNAP_AC20', 'SNAP_MI3', 'SNAP_S1',
               'SNAP_A23', 'SNAP_GR5', 'SNAP_GO15',
               'SNAP_CL15', 'SNAP_OR1', 'SNAP_VE18',
               'SNAP_PA12', 'SNAP_YE12']

    df_context_q = conv_data_competitor[(conv_data_competitor['ACCOUNT_ID'] == acc_id)]
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
        # set the width of button to num_of_cols=6 for holding the space
        num_of_cols = 6
        if len(availalbe_comp) < num_of_cols:
            dummy_list = ['aaa_' + str(i) for i in range(num_of_cols - len(availalbe_comp))]
            availalbe_comp.extend(dummy_list)

        mask_comp_dct = {'DA20': 'Vendor_Dog', 'RE4': 'Vendor_Rabbit', 'I2': 'Vendor_Iguana', 'EX1': 'Vendor_Elephant' \
            , 'ST1': 'Vendor_Salamanders', 'FI18': 'Vendor_Frog', 'TE18': 'Vendor_Turkey', 'AC20': 'Vendor_AArdvark' \
            , 'MI3': 'Vendor_Moose', 'S1': 'Vendor_Starfish', 'A23': 'Vendor_AArdvark', 'GR5': 'Vendor_Gopher','GO15': 'Vendor_Goose' \
            , 'CL15': 'Vendor_Cat', 'OR1': 'Vendor_Octopus', 'VE18': 'Vendor_Vulture', 'PA12': 'Vendor_Penguin','YE12': 'Vendor_Yaks'}

        availalbe_comp_masked = [mask_comp_dct.get(i) for i in availalbe_comp]

        pairs = zip(availalbe_comp, st.columns(len(availalbe_comp_masked)))

        for i, (comp_i, col) in enumerate(pairs):
            print(i, (comp_i, col))
            if comp_i.startswith('aaa_'): # when no competitor
                continue
            elif col.button(mask_comp_dct.get(comp_i), key=f"{comp_i}-{i}", use_container_width=True):
                for comp_con in comp_res_set[comp_i]: # there could be multiple places talking about the competitor
                    ###e5
                    comp_con_embedding_eb5 = fetch_data_competitor_context(_session=session, query=embedding_query, para_dct={'user_msg': comp_con[3:-3]} ).EMBEDED_RESULT.values[0]
                    df_retrieved_result = fetch_data_competitor_context(_session=session, query=similar_context_query,
                                                            para_dct={ 'user_msg':list(comp_con_embedding_eb5), 'acc_id':acc_id})
                    # df_retrieved_result['FLAG_DOUBLE_CHECK_COMPETITOR'] = df_retrieved_result['MEETING_CHUNK'].apply(lambda x: double_check_competitor(x, comp_i))
                    # df_retrieved_res_checked = df_retrieved_result[df_retrieved_result['FLAG_DOUBLE_CHECK_COMPETITOR']==True]

                # if df_retrieved_res_checked.shape[0]>0: -- remove the if condition
                #     context_competitor_checked = convert_multiple_pieces_to_one_feed(df_retrieved_res_checked)
                    context_competitor_checked = convert_multiple_pieces_to_one_feed(df_retrieved_result.head(3))
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

                    mask_comp_reverse_df = fetch_data_multi_args(_session=session, query=mask_comp_reverse_mapping_query)
                    mask_comp_reverse_dct = dict(zip(mask_comp_reverse_df.VENDOR, mask_comp_reverse_df['VENDOR_NICKNAMES'].apply(lambda x: eval(x))))
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
        query_question_embedding = fetch_data_multi_args_context(_session=session, query=embedding_query, para_dct={'user_msg':context_question}).EMBEDED_RESULT.values[0]
        df_retrieved_result = fetch_data_multi_args_context(_session=session, query=similar_context_query
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
    tab_task_timeline, tab_opp_nextstep_timeline = st.tabs(subtab_titles)

    with tab_task_timeline:
        df_combined_task_tmp = fetch_data_multi_args(_session = session, query=preprocessed_task_timeline_query, para_dct= {'acc_id': acc_id})

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
        df_opp_timeline_tmp = fetch_data_multi_args(_session = session, query=preprocessed_nextsteps_timeline_query, para_dct= {'acc_id': acc_id})

        if df_opp_timeline_tmp.shape[0] > 0:
            result_opp_df = rename_opp_timeline(df_opp_timeline_tmp)
            timeline_opp = st_timeline(result_opp_df[result_opp_df.start>=cutoff_date].to_dict('records'), groups=[], options={}, height="400px")
            st.write('The label of each next step is summarized using LLM based on user input of next steps in opportunity field.')
            st.write("To see details, select a tag in timeline by clicking on the label")
            st.write(timeline_opp)

        else:
            st.write('No existing Next Steps data to create opportunity timeline.')