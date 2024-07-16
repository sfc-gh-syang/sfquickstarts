# -- to set up pipeline when approved for prod: https://app.snowflake.com/sfcogsops/snowhouse_aws_us_west_2/w33skaSHB7En#query
gong_data_competitor_query = """
select conv_date
    ,	conv_time
    ,	meeting_title
    ,	conversation_key
    ,	ae_name
    ,	owner_id
    ,	access_id_lst
    ,	ae_email
    ,	account_name
    ,	theater
    ,	account_id
    ,	opp_id_lst
    ,	cleaned_dialogue
    ,	idx_databricks
    ,	idx_redshift
    ,	idx_ibm
    ,	idx_exasol
    ,	idx_starburst
    ,	idx_firebolt
    ,	idx_teradata
    ,	idx_actian
    ,	idx_microsoft
    ,	idx_sap
    ,	idx_aws
    ,	idx_greenplum
    ,	idx_google
    ,	idx_cloudera
    ,	idx_oracle
    ,	idx_vertica
    ,	idx_palantir
    ,	idx_yellowbrick
    ,	idx_microfocus
    ,	sum_idx_competitor
    ,	snap_databricks
    ,	snap_redshift
    ,	snap_ibm
    ,	snap_exasol
    ,	snap_starburst
    ,	snap_firebolt
    ,	snap_teradata
    ,	snap_actian
    ,	snap_microsoft
    ,	snap_sap
    ,	snap_aws
    ,	snap_greenplum
    ,	snap_google
    ,	snap_cloudera
    ,	snap_oracle
    ,	snap_vertica
    ,	snap_palantir
    ,	snap_yellowbrick
    ,	snap_microfocus
--from sales.activity.gong_all_conv_competitor_position
--from syang.llm_schema.gong_data_competitor
from syang.llm_schema.df_context_q 
"""

# will update it later
combined_task_query = """
select *
--from sales.activity.combined_activities_task_structured
from syang.llm_schema.df_combined_task 
"""

preprocessed_gong_timeline_query = """
select *
from syang.llm_schema.result_gong_timeline
where account_id='{acc_id}'
"""

preprocessed_nextsteps_timeline_query ="""
select *
from syang.llm_schema.RESULT_OPP_TIMELINE
where account_id='{acc_id}'
"""

combined_activities_query = """
--for task, only keep two years ago
select account_id
    , account_name
    , owner_id
    , activity_type_id_track
    , all_activities_for_acct
    , access_id_lst
--from sales.activity.combined_activities_task_gong
from syang.llm_schema.df_combined
"""

opp_processed_query = """
select ds
    , processed_date
    , account_id
    , account_name
    , opp_id
    , opp_name
    , total_acv
    , raw_opp_next_steps_str
    , cap1_close_won_opp_str
    , close_lost_opp_str
    , opp_processed_str
    , close_date
    , is_closed
    , owner_id
--from sales.activity.opp_next_steps
from syang.llm_schema.DF_PROCESSED_OPP
--where contains(access_id_lst, %s)
--and ds = (select max(ds) from sales.activity.opp_next_steps)
"""

oppty_status_query ="""
select *
--from sales.activity.dim_opportunities_status opp
from syang.llm_schema.df_oppty_status
"""

similar_context_query_ = """
select cast( {user_msg} as VECTOR(FLOAT, 1536)) as result
"""

similar_context_query_2 = """
select vector_cosine_distance(
     embedded_chunk
    , cast( {user_msg} as VECTOR(FLOAT, 768))) as score
    , emd.id
    , emd.meeting_chunk
    , emd.chunk_num
    , emd.embedded_chunk
    , emd.flag_no_embedding
    , emd.meeting_key
    , emd.meeting_date
    , emd.meeting_title
from syang.llm_schema.RESULT_TO_EMBED_EMBEDED emd
where emd.account_id = '{acc_id}'
order by score desc
"""

mistral_query = """select snowflake.cortex.complete('mistral-large',
    [
    {{'role': 'system','content': $${sys_msg}$$ }}
    , {{'role': 'user','content': $${user_msg}$$ }}
    ]
    ,{{'temperature': 0.01}}
    ):choices[0]:messages::string as result
;
"""

mistral_meddipicc_query = """select snowflake.cortex.complete('mistral-large',
    [
    {{'role': 'system','content': $${sys_msg}$$ }}
    , {{'role': 'user','content': $${user_msg}$$ }}
    ]
    ,{{'temperature': 0}}
    ):choices[0]:messages::string as result
;
"""

embedding_query = """
select snowflake.cortex.embed_text('e5-base-v2',$${user_msg}$$) as embeded_result
;
"""
