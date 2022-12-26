from snowflake.snowpark.session import Session
import json

with open('environment.json','r') as f:
    creds = json.loads(f.read())

accountname = creds['accountname']
username = creds['username']
password = creds['password']

connection_parameters = {
    "account": accountname,
    "user": username,
    "password": password,
    "role": "ACCOUNTADMIN"
}

def connect():
    try:
        session = Session.builder.configs(connection_parameters).create()
        print("connection successful!")
    except:
        raise ValueError("error while connecting with db")
    return session