conv_data_competitor_query = """
select *
from syang.llm_schema.df_context_competitor_conv
"""

combined_activities_query = """
select account_id
    , account_name
    , owner_id
    , activity_type_id_track
    , all_activities_for_acct
from syang.llm_schema.df_combined_activities
where account_id='{acc_id}'
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
from syang.llm_schema.df_processed_opp
where account_id='{acc_id}'
"""

preprocessed_task_timeline_query = """
select *
from syang.llm_schema.result_task_timeline 
where account_id='{acc_id}'
"""

preprocessed_conv_timeline_query = """
select *
from syang.llm_schema.result_conv_timeline
where account_id='{acc_id}'
"""

preprocessed_nextsteps_timeline_query = """
select *
from syang.llm_schema.result_opp_timeline
where account_id='{acc_id}'
"""

mask_comp_reverse_mapping_query = """
select * 
from syang.llm_schema.mask_comp_reverse_mapping
"""

similar_context_query = """
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

embedding_query = """
select snowflake.cortex.embed_text('e5-base-v2',$${user_msg}$$) as embeded_result
;
"""
