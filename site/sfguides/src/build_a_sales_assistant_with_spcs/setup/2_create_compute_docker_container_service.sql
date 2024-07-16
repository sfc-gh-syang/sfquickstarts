--example db, schema used
--use database syang;
--use schema serve_schema;

/*1. CREATE  COMPUTE POOL*/
show compute pools;

create compute pool syang_gpum
  min_nodes = 1
  max_nodes = 2
  auto_resume = true
  initially_suspended = true
  instance_family = gpu_nv_m
  auto_suspend_secs = 3600
;
alter compute pool syang_gpum resume;


/*2. BUILD THE DOCKER IMAGE, AND HAVE THE IMAGE REGISTERED IN REPO*/
show image repositories;

-- show dockers
-- call SYSTEM$REGISTRY_LIST_IMAGES('/syang/serve_schema/serve_repo');


/*3. CREATE CONTAINER SERVICE ON THE COMPUTE POOL*/
create service streamlit_multi_demo
in compute pool syang_gpum
from @specs
spec='streamlit_multi.yaml'
min_instances = 1
max_instances = 4
external_access_integrations = (HF_ACCESS_INTEGRATION)
;

alter service streamlit_multi_demo resume;
alter service streamlit_multi_demo suspend;
drop service streamlit_multi_demo;

-- check the yaml file of the service
-- ls @serve_schema.specs;

/*4. CHECK THE ENDPOINTS, WHEN STREAMLIT SERVICE IS READY*/
select system$get_service_status('streamlit_multi_demo');
show endpoints in service streamlit_multi_demo;

