-- MUST BE RUN BY ACCOUNTADMIN to allow connecting to huggingface to download the model
CREATE OR REPLACE NETWORK RULE hf_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('huggingface.co', 'cdn-lfs.huggingface.co','cdn-lfs-us-1.huggingface.co');

CREATE EXTERNAL ACCESS INTEGRATION hf_access_integration
  ALLOWED_NETWORK_RULES = (hf_network_rule)
  ENABLED = true;

GRANT USAGE ON INTEGRATION hf_access_integration TO ROLE <your_role>;
GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE <your_role>;


-- Stage and impage repository to store LLM models
use schema serve_schema;

CREATE STAGE IF NOT EXISTS models
 DIRECTORY = (ENABLE = TRUE)
 ENCRYPTION = (TYPE='SNOWFLAKE_SSE');

-- Stage to store yaml specs
CREATE STAGE IF NOT EXISTS specs
 DIRECTORY = (ENABLE = TRUE)
 ENCRYPTION = (TYPE='SNOWFLAKE_SSE');

-- Image registry
CREATE IMAGE REPOSITORY if not exists serve_repo;
