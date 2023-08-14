import json 
import datetime
import os

from bs4 import BeautifulSoup
import re

def remove_html_formatting(input_string):
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(input_string, "html.parser")
    cleaned_text = soup.get_text(separator=" ")
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def clean_intercom(data_final,path):
    intercom=[]
    intercom_json_meta={}
    i=1

    for data in data_final:
      try:
        intercom_json_meta["source"]="intercom"
        intercom_json_meta["created time"]=str(datetime.datetime.fromtimestamp(data["created_at"]))
        intercom_json_meta["updated time"]=str(datetime.datetime.fromtimestamp(data["updated_at"]))
        intercom_json_meta[f"conversation type"]= data['type']
        intercom_json_meta["id"]=data["id"]
        intercom_json_meta[f"source delivered_as"]= data['source']['delivered_as']
        intercom_json_meta[f"source subject"]= remove_html_formatting(data['source']['subject'])
        intercom_json_meta[f"source body"]= remove_html_formatting(data['source']['body'])
        if data['source']['author'] is not None:
            intercom_json_meta[f"source author type"]= data['source']['author']['type']
            intercom_json_meta[f"source author name"]= data['source']['author']['name']
        try:
            intercom_json_meta[f"conversation rating"]= str(data['conversation_rating']['rating'])
        except Exception as e:
            pass
        intercom_json_meta[f"status"]= data['state']
        intercom_json_meta[f"priority"]= data['priority']

        intercom_json_meta["conversation"]= []
        complete_conversation=""
        for c_item in data["conversation_parts"]["conversation_parts"]:
            intercom_j={}
            if str(c_item['body'])!="None":
                intercom_j[f"message body"]= remove_html_formatting(str(c_item['body']))
                complete_conversation=complete_conversation+remove_html_formatting(str(c_item['body']))
                intercom_j[f"message author"]= c_item['author']['name']
                intercom_json_meta['conversation'].append(intercom_j)

        intercom_json_meta["complete conversation"]=complete_conversation
        intercom.append(intercom_json_meta)
      except:
        pass

    json_object = json.dumps(intercom, indent=4)
    os.makedirs('./'+str(path),exist_ok=True)
    filename = './'+str(path)+"intercom.json"
    with open(filename, "a") as outfile:
        outfile.write(json_object)
    return intercom,filename

def clean_jira(data,path): ###Edit required for multiple id
    jira=[]
    jira_meta={}
    jira_meta["total issues"]= data['total']
    jira_meta["source"]="jira"
    jira.append(jira_meta)
    for item in data["issues"]:
        jira_json={}
        jira_json[f"issue id"]= item['id']
        jira_json[f"issue created"]= item['fields']['created']
        jira_json[f"issue duedate"]= item['fields']['duedate']
        jira_json[f"issue type"]= item['fields']['issuetype']['name']
        jira_json[f"issue assignee"]= item['fields']['assignee']['displayName']
        jira_json[f"issue creator"]= item['fields']['creator']['displayName']
        jira_json[f"issue reporter"]= item['fields']['reporter']['displayName']

        if item['fields']['labels']!=[]:
            jira_json[f"issue label"]=item['fields']['labels']
        if item['fields']['status'] is not None:
            description= item['fields']['status']['description']
        if item['fields']['resolution'] is not None:
            description= description+ item['fields']['resolution']['description']
        if item['fields']['issuetype'] is not None:
            description=description+ item['fields']['issuetype']['description']

        jira_json[f"issue description"]= description
        text=""

        for ct_item in item['fields']["description"]["content"]:
            for cnt_item in ct_item["content"]:
                try:
                    if len(cnt_item['text'])>1:
                        text=text+cnt_item['text']+" "
                except:
                    pass

        jira_json[f"issue body"]= text
        jira_json[f"issue summary"]= item['fields']['summary']
        jira_json[f"issue priority"]= item['fields']['priority']['name']
        jira_json[f"issue status"]= item['fields']['status']['name']
        jira.append(jira_json)
        
    json_object = json.dumps(jira, indent=4)
    os.makedirs('./'+str(path),exist_ok=True)
    filename = './'+str(path)+"jira.json"
    with open(filename, "a") as outfile:
        outfile.write(json_object)
    return jira,filename

def clean_meet(data,path):
    sorted_json=[]
    participants={}

    for item in data["attendeeslist"]:
        participants[item["userId"]]=item["displayName"]

    metadata={}
    metadata["meetingtitle"]=data["meetingtitle"]
    metadata["source"]="meet"
    metadata["companyid"]=data["companyid"]
    metadata["callduration"]=data["callduration"]
    metadata["completetranscription"]=data["completetranscription"]
    sorted_json.append(metadata)

    for item in data["transcription"]:
        transcription={}
        transcription["transcript"]=item["results"][0]["alternatives"][0]["transcript"]
        transcription["speaker"]=participants[item["results"][0]["track"]]
        transcription["user_id"]=item["results"][0]["track"]
        sorted_json.append(transcription)
        
    json_object = json.dumps(sorted_json, indent=4)
    os.makedirs('./'+str(path),exist_ok=True)
    filepath = './'+str(path)+"meet.json"
    with open(filepath, "a") as outfile:
        outfile.write(json_object)
    return sorted_json, filepath

def clean_zoom(data,path):
    sorted_json=[]
    participants={}

    for item in data["attendeeslist"]:
        participants[item["userId"]]=item["displayName"]

    metadata={}
    metadata["meetingtitle"]=data["meetingtitle"]
    metadata["companyid"]=data["companyid"]
    metadata["source"]="zoom"
    metadata["callduration"]=data["callduration"]
    metadata["completetranscription"]=data["completetranscription"]
    sorted_json.append(metadata)

    for item in data["transcription"]:
        transcription={}
        transcription["transcript"]=item["results"][0]["alternatives"][0]["transcript"]
        transcription["speaker"]=participants[item["results"][0]["track"]]
        transcription["user_id"]=item["results"][0]["track"]
        sorted_json.append(transcription)

    json_object = json.dumps(sorted_json, indent=4)
    os.makedirs('./'+str(path),exist_ok=True)
    filepath = './'+str(path)+"meet.json"
    with open(filepath, "a") as outfile:
        outfile.write(json_object)
    return sorted_json, filepath