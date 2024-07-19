import os
import streamlit as st
import snowflake.connector
# from snowflake.connector.pandas_tools import write_pandas
from snowflake.snowpark import Session

from dotenv import load_dotenv
load_dotenv()


@st.cache_resource()
def create_session_object():
    if not os.path.exists("/snowflake/session/token"):
        connection_parameters = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USERNAME"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
            "client_session_keep_alive":  True,
            "network_timeout": 120
        }
        session = Session.builder.configs(connection_parameters).create()
    else:
        with open("/snowflake/session/token", "r") as f:
            token = f.read()
        connection_parameters = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "host": os.getenv("SNOWFLAKE_HOST"),
            "authenticator": "oauth",
            "token": token,
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        }
        session = Session.builder.configs(connection_parameters).create()
    return session

session = create_session_object()


@st.cache_data(ttl="3600", show_spinner="Fetching data...")
def fetch_data_multi_args(_session,
                           query=None,
                           para_dct={},
                           ):
    """
    Executes the query and return a dataframe
    @query: input query
    @para: parse account selected
    """
    if para_dct != {}:
        try:
            df_sql = _session.sql(query.format(**para_dct)).to_pandas()
            return df_sql
        except Exception as e:
            st.error("Query Timeout. Please go to Home page and retry later: {}".format(e))
            st.stop()
    else:
        try:
            df_sql = _session.sql(query).to_pandas()
            return df_sql
        except Exception as e:
            st.error("Query Timeout. Please go to Home page and retry later: {}".format(e))
            st.stop()



@st.cache_resource(show_spinner="Running LLM to get answer ...")
def fetch_data_multi_args_general(_session,
                                  query=None,
                                  para_dct={},
                                  ):
    """
    Executes the query and return a dataframe
    @query: input query
    @para: parse account selected
    """
    if para_dct != {}:
        try:
            df_sql = _session.sql(query.format(**para_dct)).to_pandas()
            return df_sql
        except Exception as e:
            st.error("Query Timeout. Please go to Home page and retry later: {}".format(e))
            st.stop()
    else:
        try:
            df_sql = _session.sql(query).to_pandas()
            return df_sql
        except Exception as e:
            st.error("Query Timeout. Please go to Home page and retry later: {}".format(e))
            st.stop()



@st.cache_resource(ttl="9000",  show_spinner="Running LLM to get answer...")
def fetch_data_competitor_context(_session,
                                   query=None,
                                   para_dct={},
                                   ):
    """
    Executes the query and return a dataframe
    @query: input query
    @para: parse account selected
    """
    if para_dct != {}:
        try:
            df_sql = _session.sql(query.format(**para_dct)).to_pandas()
            return df_sql
        except Exception as e:
            st.error("Query Timeout. Please go to Home page and retry later: {}".format(e))
            st.stop()
    else:
        try:
            df_sql = _session.sql(query).to_pandas()
            return df_sql
        except Exception as e:
            st.error("Query Timeout. Please go to Home page and retry later: {}".format(e))
            st.stop()



@st.cache_resource(ttl="9000",  show_spinner="Running LLM to get answer ...")
def fetch_data_multi_args_context(_session,
                                   query=None,
                                   para_dct={},
                                   ):
    """
    Executes the query and return a dataframe
    @query: input query
    @para: parse account selected
    """
    if para_dct != {}:
        try:
            df_sql = _session.sql(query.format(**para_dct)).to_pandas()
            return df_sql
        except Exception as e:
            st.error("Query Timeout. Please go to Home page and retry later: {}".format(e))
            st.stop()
    else:
        try:
            df_sql = _session.sql(query).to_pandas()
            return df_sql
        except Exception as e:
            st.error("Query Timeout. Please go to Home page and retry later: {}".format(e))
            st.stop()



@st.cache_resource(show_spinner="Fetching data...")
def fetch_data_multi_args_meddpic(_session,
                                   query=None,
                                   para_dct={},
                                   ):
    """
    Executes the query and return a dataframe
    @query: input query
    @para: parse account selected
    """
    if para_dct != {}:
        try:
            df_sql = _session.sql(query.format(**para_dct)).to_pandas()
            return df_sql
        except Exception as e:
            st.error("Query Timeout. Please go to Home page and retry later: {}".format(e))
            st.stop()
    else:
        try:
            df_sql = _session.sql(query).to_pandas()
            return df_sql
        except Exception as e:
            st.error("Query Timeout. Please go to Home page and retry later: {}".format(e))
            st.stop()