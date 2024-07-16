# Welcome to the demostration of Sales Assistant on Snowpark Container Service
Author: Stella Yang

As SALES ASSISTANT is developed based on LLM model, a one-stop-shop for sales people to grab the key points about client.

Refer [Customized AI w/ Snowpark Container Services](https://docs.google.com/presentation/d/1ShK4KTQsTfJSRkKyViaxY8OamlGBdKEl/edit?usp=drive_link&ouid=112777878365644760549&rtpof=true&sd=true) for general info.

## Pre-requisites
- Docker Desktop (or some way to build containers)
- Access to Snowpark Container Services - details on region availability can be found [here](https://github.com/Snowflake-Labs/spcs-updates)
- A [HuggingFace](https://huggingface[create_objects.sql](setup%2Fcreate_objects.sql).co/) account

## Setup for Demo

### Clone this sample

`git clone https://github.com/snowflakedb/datascience-airflow/tree/summit-demo/sales/etl/summit_demo`

### Create necessary objects

Browse to [setup/1_create_necessary_objects.sql](./setup/1_create_necessary_objects.sql) and run the SQL to create nessesary objects, use ACCOUNTADMIN role to run these.
[//]: # (The SECURITY INTEGRATION and NETWORK / INTEGRATIONS **MUST** be run by an ACCOUNTADMIN for the account &#40;if already run for the account this can be skipped&#41;.)

This contains:
1. A stage to store the LLM that is downloaded.
2. stage to upload and store the YAML specs for the containers.
3. container image registry to push our docker images


[//]: # (1. SECURITY INTEGRATION to allow Oauth to containers &#40;if not already enabled for the account&#41;, and NETWORK access to allow the container to download from huggingface.)

### Create compute pool, register image, create container service 
Browse to [setup/2_create_compute_docker_container_service.sql](./setup/2_create_compute_docker_container_service.sql). Note that you need to use a non-ACCOUNTADMIN role to create services.
This contains:
1. Compute pool to run the containers. 
2. Check the image registry created in setup_objects
3. Build the container image and check image registered in repo
4. Create the container service in the compute pool
5. Access the endpoint of the container service
Out of this, the prerequisite of step2, step3 and step4 will be instructed in below section.

#### Create compute pool and check creation by running `SHOW COMPUTE POOLS;`
![image](https://github.com/user-attachments/assets/7e660b3c-f3df-4703-a06e-2d8fd5ffacd4)

#### Login to the Snowflake Image Registry
Get the registry URL for the created registry by running `SHOW IMAGE REPOSITORIES` in the schema where you ran the above setup scripts. Copy the `repository_url` value, for your docker image registry later. 
![image](https://github.com/user-attachments/assets/7f401ad7-ec7f-4ed1-a92d-a0ab9df8e1d6)

#### Build the container image
The container you are going to example is the last container that we will use serve the streamlit app
```commandline
syang@XH0GL74PFW summit_demo % docker build --rm --platform linux/amd64 -t streamlit_multi .         
syang@XH0GL74PFW summit_demo % docker tag streamlit_multi sfengineering-ml4snow.registry.snowflakecomputing.com/syang/serve_schema/serve_repo/streamlit_multi
syang@XH0GL74PFW summit_demo % docker push sfengineering-ml4snow.registry.snowflakecomputing.com/syang/serve_schema/serve_repo/streamlit_multi               
```
<img width="842" alt="image" src="https://github.com/user-attachments/assets/230b9fb9-96e5-4a1f-8757-a4a810dba787">
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/a0a91e76-dfbb-4ae0-9525-6c78b4222abd">

check image registered in repo by running this line`call SYSTEM$REGISTRY_LIST_IMAGES( '/syang/serve_schema/serve_repo')`.
#### Create the container service
```commandline
create service streamlit_multi_demo
in compute pool syang_gpum
from @specs
spec='streamlit_multi.yaml'
min_instances = 1
max_instances = 4
external_access_integrations = (HF_ACCESS_INTEGRATION)
;
```
