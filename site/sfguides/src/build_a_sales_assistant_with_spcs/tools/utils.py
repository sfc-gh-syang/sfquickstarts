import json
import os
import pandas as pd
import numpy as np
import streamlit as st
from st_aggrid import JsCode, GridOptionsBuilder, AgGrid, GridUpdateMode, ColumnsAutoSizeMode
from tools.sales_assistant_llm_query import *
from tools.sales_assistant_prompts import *
from tools.load_data_query import *


# configure selection
cellstyle_jscode_gong = JsCode("""
function(params){
    if (params.value == 'In Pursuit') {
        return {
            'color': 'black',
            'backgroundColor': '#29B5E8',
        }
    }
    if (params.value == 'Implementation') {
        return{
            'color': 'black',
            'backgroundColor': '#11567F',
        }
    }
    if (params.value == 'Production') {
        return{
            'color': 'black',
            'backgroundColor': '#8A999E',
        }
    }
}
""")



def add_colored_header(header,
                       color_name="snowflakeblue-100",
                       description=None
                       ):
    """Adds a page title with a colored underline and an optional description."""
    color= '#29B5E8'# get_hex_color(color_name)
    st.markdown(f"## {header}")
    st.write(
        f'<hr style="background-color: {color}; margin-top: 0; margin-bottom: 0; height: 6px; border: none; border-radius: 6px;">',
        unsafe_allow_html=True
    )
    if description:
        st.caption(description)



def setup_form_universal(input_preselect=None,
                        default_selection=None,
                        question_name=''
               ):
    """
    This function sets up form to enter a question or select a pre-defined question
    Question entered always takes priority unless it's cleared out
    """

    with st.form('question_form'+question_name):
        text_input = st.text_area(
            label=f"Enter your own question:",
            value="",
            height=30,
            max_chars=255)
        text_select = st.selectbox(label="Or select from the pre-listed questions (clear the question entered above before submit)",
                                   options=input_preselect)
        submitted = st.form_submit_button('Submit')
        st.caption("Sales Assistant can make mistakes. Consider double checking important information.")

        # update session state after form submission
        if submitted:
            st.session_state["text_input"+question_name] = text_input
            st.session_state["text_select"+question_name] = text_select

    # st.write('st.session_state["text_input"]=',st.session_state["text_input"])
    if st.session_state["text_input"+question_name] and st.session_state["text_input"+question_name]!='':
        selection = st.session_state["text_input"+question_name]
    elif st.session_state["text_select"+question_name] != default_selection[0]:
        selection = st.session_state["text_select"+question_name]
    else:
        selection = "No question"
    return selection



def cleanup_client_name(df = None
                        , cols = []
                        , old_name_lst=[]
                        , new_name_lst=[]):
    for col in cols:
        print(col)
        for old_i, new_i in zip(old_name_lst,new_name_lst):
            # print(old_i, new_i)
            df[col] =df[col].str.replace(old_i,new_i)
    return df



def combine_context_and_quesiton_to_prompt(prompt='', question='', context_i='', today_date=''):

    # dct = {"general_question": general_question
    #        , "context_i": context_i}
    dct = {"question": question
           , "context_i": context_i
           , "today_date": today_date}

    prompt = prompt.format(**dct)

    return prompt



def get_unique_competitor_context_set(comp_res):
    """
    Combines and extracts unique competitors from a list of competitor results.
    """
    agg_dct = {}
    # print(comp_res)
    # print(len(comp_res))
    # ['*'+i+'*' for i in comp_res]
    for dct_i in comp_res:
        print(dct_i)
        if dct_i!={}:
            for key in dct_i.keys():
                print('working on key=', key)
                if key in agg_dct.keys():
                    # print('dct_i', dct_i)
                    print('already existing a key')
                    # there is a syntax error here
                    print('agg_dct[key] type', type(agg_dct[key]) )
                    tmp_append_res = agg_dct[key]
                    tmp_append_res.append(dct_i[key])
                    # print('dct_i[key]=',dct_i[key])
                    print('agg_dct[key]=', tmp_append_res)
                    agg_dct[key] = tmp_append_res
                else:
                    agg_dct[key] = [dct_i[key]]
                    # print('CHECK EXISTING AGG_DCT:',agg_dct)
    return agg_dct



def create_competitor_context(cols):
    res = {}

    DATABRICKS = cols[0]
    REDSHIFT = cols[1]
    IBM = cols[2]
    EXASOL = cols[3]
    STARBURST = cols[4]
    FIREBOLT = cols[5]
    TERADATA = cols[6]
    ACTIAN = cols[7]
    MICROSOFT = cols[8]

    SAP = cols[9]
    AWS = cols[10]
    GREENPLUM = cols[11]
    GOOGLE = cols[12]
    CLOUDERA = cols[13]

    ORACLE = cols[14]
    VERTICA = cols[15]
    PALANTIR = cols[16]
    YELLOWBRICK = cols[17]

    if DATABRICKS!='':
        res['DATABRICKS']=DATABRICKS
    if REDSHIFT!='':
        res ['REDSHIFT']= REDSHIFT
    if IBM !='':
        res ['IBM']= IBM
    if EXASOL !='':
        res ['EXASOL']= EXASOL
    if STARBURST !='':
        res ['STARBURST']= STARBURST
    if FIREBOLT !='':
        res ['FIREBOLT']= FIREBOLT
    if TERADATA !='':
        res ['TERADATA']= TERADATA
    if ACTIAN !='':
        res ['ACTIAN']= ACTIAN
    if MICROSOFT !='':
        res ['MICROSOFT']= MICROSOFT
    if SAP !='':
        res ['SAP']= SAP
    if AWS !='':
        res ['AWS']= AWS
    if GREENPLUM !='':
        res ['GREENPLUM']= GREENPLUM
    if GOOGLE !='':
        res ['GOOGLE']= GOOGLE
    if CLOUDERA !='':
        res ['CLOUDERA']= CLOUDERA
    if ORACLE !='':
        res ['ORACLE']= ORACLE
    if VERTICA !='':
        res ['VERTICA']= VERTICA
    if PALANTIR !='':
        res ['PALANTIR']= PALANTIR
    if YELLOWBRICK !='':
        res ['YELLOWBRICK']= YELLOWBRICK
    # if res =='':
    #     res = 'No competitor'
    # if res.endswith(', ') == True:
    #     res = res[:-2]
    return res



def double_check_competitor(chunk,comp_i):
    competitor_dct = {'DATABRICKS':['databrick','data brick']
        ,'REDSHIFT':['redshift','red shift']
        ,'IBM':['ibm']
        ,'EXASOL':['exasol']
        ,'STARBURST':['starburst']
        ,'FIREBOLT':['firebolt', 'fire bolt']
        ,'TERADATA':['teradata','tera data']
        ,'ACTIAN':['actian']
        ,'MICROSOFT':['microsoft','azure','synapse','fabric']
        ,'SAP':['sap']
        ,'AWS':['aws','aws emr']
        ,'GREENPLUM':['greenplum']
        ,'GOOGLE':['google','gcp','big query']
        ,'CLOUDERA':['cloudera']
        ,'ORACLE':['oracle']
        ,'VERTICA':['vertica']
        ,'PALANTIR':['vertica']
        ,'YELLOWBRICK':['yellowbrick', 'yellow brick']
    }
    flag = 0
    for comp_i in competitor_dct.get(comp_i):
        if chunk.lower().__contains__(comp_i):
            flag+=1
    return flag>0



def convert_multiple_pieces_to_one_feed(df):
    """
    Converts multiple pieces of retrieved results from a DataFrame into a combined feed.

    Parameters:
    - df (DataFrame): Pandas DataFrame containing retrieved results with columns like 'MEETING_DATE', 'MEETING_CHUNK', etc.

    Returns:
    - str: A combined feed containing information from multiple pieces, sorted by meeting dates.
    """
    combined_pieces = ''
    df['MEETING_DATE_STR'] = df['MEETING_DATE'].astype(str)
    for i in df.index:
        print(i)
        if df.shape[0]>1:
            piece_i_dict = dict(df.loc[i])  # wired thing should use iloc to access by index, but only loc works for acc_id='0010Z00001xwhYJQAY'
        else:
            piece_i_dict = dict(df)
        # print('------start the piece------------')
        # combined_pieces = combined_pieces +'\n\nslice '+str(i)+':\n' +  'Conversation date: '+ piece_i_dict['metadata']['source'].split('__*__')[1]+'\n' + piece_i_dict['page_content'].replace('Company_ABC_','Snowflake')
        combined_pieces = combined_pieces + '\n\nslice ' + str(i) + ':\n' + 'Conversation date: ' + piece_i_dict['MEETING_DATE_STR'] + '\n' + piece_i_dict['MEETING_CHUNK'].replace('Company_ABC_', 'Snowflake')

    return combined_pieces



def convert_num_to_string_flexible(col):
    """
    format conversion to M, with - sign reallocated to proper position
    as observed most of case positive do not need sign in table
    :param col: a float
    :return: a string
    """
    num = col

    if np.isnan(num):
        return 'NA'

    sign = num>=0
    if abs(num)>= 1000000000:
        num_in_mm = num/1000000000
        if sign:
            str_in_mm = '$' + str("{:,}".format(round(num_in_mm,2))) + 'B'
        else:
            str_in_mm = '-$' + str("{:,}".format(round( abs(num_in_mm),2))) + 'B'

        return str_in_mm

    elif abs(num)>= 1000000:
        num_in_mm = num/1000000
        if sign:
            str_in_mm = '$' + str("{:,}".format(round(num_in_mm,2))) + 'M'
        else:
            str_in_mm = '-$' + str("{:,}".format(round( abs(num_in_mm),2))) + 'M'

        return str_in_mm

    elif abs(num)>= 1000:
        num_in_k = num/1000
        if sign:
            str_in_k = '$' + str("{:,}".format(round(num_in_k,1))) + 'K'
        else:
            str_in_k = '-$' + str("{:,}".format(round(abs(num_in_k),1))) + 'K'
        return str_in_k

    elif abs(num) < 1000:
        if sign:
            str_in_1 = '$' + str("{:,}".format(int(num)))
            if str_in_1 == '$0.0':
                str_in_1 = '$0'
        else:
            str_in_1 = '-$' + str("{:,}".format(int(abs(num))))
        return str_in_1

    else:
        if np.isnan(num):
            return '$0'
        else:
            str_in_1 = '$' + str("{:,}".format(int(num)))

        return str_in_1




def convert_processed_opp_to_str(df = None):
    """
    Converts processed opportunity data from a DataFrame to a formatted string.

    Parameters:
    - df (DataFrame): Pandas DataFrame containing processed opportunity data with columns like 'OPP_NAME', 'TOTAL_ACV_', and 'OPP_PROCESSED_STR'.

    Returns:
    - str: A formatted string containing information about processed opportunities, including OPP_PROCESSED_STR, sorted by CLOSE_DATE.
    """
    df.sort_values(['CLOSE_DATE'], inplace=True)
    cnt=0
    res_str = 'NEXT STEPS section:'
    for idx in df.index:
        cnt+=1
        # print(idx)
        if df.loc[idx]['TOTAL_ACV_']!='$0':
            res_str+= '\nSlice '+ str(cnt) + ', Opporutnity '+df.loc[idx]['OPP_NAME'] +' with Total ACV'+df.loc[idx]['TOTAL_ACV_']+':\n'+df.loc[idx]['OPP_PROCESSED_STR']
        else:
            res_str += '\nSlice ' + str(cnt) + ', Opporutnity ' + df.loc[idx]['OPP_NAME'] + ':\n' + df.loc[idx]['OPP_PROCESSED_STR']
    return res_str



# can skip if uploaded pre-process
def convert_processed_opp_separate_to_str(df = None):
    """
    Converts processed opportunity data from a DataFrame to a formatted string, for single oppty

    Parameters:
    - df (DataFrame): Pandas DataFrame containing processed opportunity data with columns like 'OPP_NAME', 'TOTAL_ACV_', and 'OPP_PROCESSED_STR'.

    Returns:
    - str: A formatted string containing information about processed opportunities, including OPP_PROCESSED_STR, sorted by CLOSE_DATE.
    """
    df.sort_values(['CLOSE_DATE'], inplace=True)
    cnt=0
    res_str = ''
    for idx in df.index:
        cnt+=1
        # if df.loc[idx]['TOTAL_ACV_']!='$0':
        #     res_str+= '\nSlice '+ str(cnt) + ', Opporutnity '+df.loc[idx]['OPP_NAME'] +' with Total ACV'+df.loc[idx]['TOTAL_ACV_']+':\n'+df.loc[idx]['OPP_PROCESSED_STR']
        # else:
        res_str += '\nSlice ' + str(cnt) + ':\nOPP_NAME: ' + df.loc[idx]['OPP_NAME'] + ':\n' + df.loc[idx]['OPP_PROCESSED_STR']
    return res_str
    # return '\nSlice ' + str(cnt) + ':\nOPP_NAME: ' + df['OPP_NAME'] + ':\n' + df['OPP_PROCESSED_STR']



def convert_processed_opp_to_str(df = None):
    """
    Converts processed opportunity data from a DataFrame to a formatted string.

    Parameters:
    - df (DataFrame): Pandas DataFrame containing processed opportunity data with columns like 'OPP_NAME', 'TOTAL_ACV_', and 'OPP_PROCESSED_STR'.

    Returns:
    - str: A formatted string containing information about processed opportunities, including OPP_PROCESSED_STR, sorted by CLOSE_DATE.
    """
    df.sort_values(['CLOSE_DATE'], inplace=True)
    cnt=0
    res_str = 'NEXT STEPS section:'
    for idx in df.index:
        cnt+=1
        # print(idx)
        if df.loc[idx]['TOTAL_ACV_']!='$0':
            res_str+= '\nSlice '+ str(cnt) + ', Opporutnity '+df.loc[idx]['OPP_NAME'] +' with Total ACV'+df.loc[idx]['TOTAL_ACV_']+':\n'+df.loc[idx]['OPP_PROCESSED_STR']
        else:
            res_str += '\nSlice ' + str(cnt) + ', Opporutnity ' + df.loc[idx]['OPP_NAME'] + ':\n' + df.loc[idx]['OPP_PROCESSED_STR']
    return res_str


def convert_raw_opp_to_context_str(df = None):
    """
    Converts raw opportunity data from a DataFrame to a formatted string.

    Parameters:
    - df (DataFrame): Pandas DataFrame containing opportunity data with columns like 'OPP_NAME', 'IS_CLOSED', 'IS_WON', 'IS_CAP_ONE',
                     'CLOSE_DATE_1Y_PRIOR', 'CLOSE_DATE', and 'NEXT_STEPS'.

    Returns:
    - str: A formatted string containing information about opportunities, including NEXT_STEPS, sorted by CLOSE_DATE.
    """

    df['CLOSE_DATE'] = df['CLOSE_DATE'].astype(str)
    df['CLOSE_DATE_1Y_PRIOR'] = df['CLOSE_DATE_1Y_PRIOR'].astype(str)
    df.sort_values(['CLOSE_DATE'], inplace=True)
    df.reset_index( inplace=True)

    count = 0
    combined_opps = '\n\n'
    cap1_close_won_opp = '\n\n'

    for idx in df.index:
        if df.loc[idx]['IS_CLOSED'] and df.loc[idx]['IS_WON'] and df.loc[idx]['IS_CAP_ONE']:
            combined_opps = combined_opps + '\nOpportunity ' + str(count) +' '+df.loc[idx]['OPP_NAME'] \
                            + ' has NEXT_STEPS between CLOSE_DATE_1Y_PRIOR ' + df.loc[idx]['CLOSE_DATE_1Y_PRIOR']\
                            + ' and CLOSE_DATE ' + df.loc[idx]['CLOSE_DATE'] + ':'\
                            + '\n' + df.loc[idx]['CLOSE_DATE'] +' - Get Cap 1 CLOSED WON. ' + df.loc[idx]['NEXT_STEPS'].replace('Company_ABC_', 'Snowflake').replace('\r\n',' ')
            cap1_close_won_opp = '\n\nCAP1_CLOSED_WON section:\n\nOn ACTIVITY_DATE ' + df.loc[idx]['CLOSE_DATE'] +' -Get Cap 1 CLOSED WON.'

        else:
            combined_opps = combined_opps + '\nOpportunity ' + str(count) +' '+df.loc[idx]['OPP_NAME'] \
                            + ' has NEXT_STEPS between CLOSE_DATE_1Y_PRIOR ' + df.loc[idx]['CLOSE_DATE_1Y_PRIOR']\
                            + ' and CLOSE_DATE ' + df.loc[idx]['CLOSE_DATE'] + ':'\
                            + '\n' + df.loc[idx]['NEXT_STEPS'].replace('Company_ABC_', 'Snowflake').replace('\r\n',' ')
        count += 1

    return combined_opps, cap1_close_won_opp



def combine_context_to_prompt(prompt, context_i):
    dct = {"context_i": context_i}
    prompt = prompt.format(**dct)

    return prompt



def rename_opp_timeline(result_opp_raw):
    result_opp_raw.rename(columns={'START': 'start'
                        , 'CONTENT': 'content'
                        , 'TITLE': 'title'
                        , 'OPPORTUNITY_NAME':'opportunity name'
                        }, inplace=True)
    return result_opp_raw[['opportunity name', 'start', 'content', 'title']]



def _rename_gong_timeline_df(df=None):
    df.rename(columns={'MEETING_TITLE': 'Meeting Title'
            , 'ACTIVITY_DATE': 'Meeting Date'
            , 'OWNER_FUNCTION': 'Owner Function'
            , 'TAG': 'Tag'
            , 'MEETING_BRIEF':'Brief'
            , 'MEETING_ACTIONS':'Follow-up Actions'}

            , inplace=True)
    df.sort_values('Meeting Date', inplace=True)
    return df



def gen_grid_builder_for_gong_df(result_gong_raw):
    result_gong_df = _rename_gong_timeline_df(df=result_gong_raw)
    gb_gong_timeline = GridOptionsBuilder.from_dataframe(result_gong_df[['Meeting Title'
        , 'Meeting Date'
        , 'Owner Function'
        , 'Tag']])

    gb_gong_timeline.configure_columns(result_gong_df, cellStyle=cellstyle_jscode_gong)
    gb_gong_timeline.configure_selection(selection_mode="single", use_checkbox=True)
    gb_gong_timeline.configure_side_bar()

    data_gong_timeline = AgGrid(result_gong_df,
                                gridOptions=gb_gong_timeline.build(),
                                enable_enterprise_modules=True,
                                allow_unsafe_jscode=True,
                                update_mode=GridUpdateMode.SELECTION_CHANGED,
                                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW)
    return data_gong_timeline



@st.cache_resource
def load_and_create_text2image_model_pipe():
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import torch

    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_unet.safetensors"  # Use the correct ckpt for your step setting

    cache_dir = "../stage/diffusers/"
    unet_config = UNet2DConditionModel.load_config(base, cache_dir=cache_dir, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config, subfolder="unet").to("cuda", torch.float16)

    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))

    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(
        "cuda")
    return pipe