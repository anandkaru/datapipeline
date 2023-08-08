import json 
import datetime
import os

def clean_intercom(data,path):  ###Edit required for multiple
    intercom=[]
    intercom_json_meta={}
    i=1

    intercom_json_meta["source"]="intercom"
    intercom_json_meta["time"]=str(datetime.datetime.now())
    intercom_json_meta[f"conversation {i} type"]= data['type']
    intercom_json_meta["id"]=data["id"]
    intercom_json_meta[f"source delivered_as"]= data['source']['delivered_as']
    intercom_json_meta[f"source subject"]= data['source']['subject']
    intercom_json_meta[f"source body"]= data['source']['body']
    if data['source']['author'] is not None:
        intercom_json_meta[f"source author type"]= data['source']['author']['type']
        intercom_json_meta[f"source author name"]= data['source']['author']['name']
        # intercom_json_meta[f"source author email"]= data['source']['author']['email']

    intercom_json_meta[f"status"]= data['state']
    # intercom_json_meta[f"read"]= str(data['read'])
    intercom_json_meta[f"priority"]= data['priority']
    if data['conversation_rating'] is not None:
        intercom_json_meta[f"conversation {i} conversation rating"]= str(data['conversation_rating']['rating'])
        # intercom_json_meta[f"conversation {i} conversation remark"]= str(data['conversation_rating']['remark'])
        # intercom_json_meta[f"conversation {i} conversation contact type"]= data['conversation_rating']['contact']['type']
    intercom.append(intercom_json_meta)

    for c_item in data["conversation_parts"]["conversation_parts"]:
        intercom_json={}
        # intercom_json[f"conversation {i} conversation part {j} part_type"]= c_item['part_type']
        intercom_json[f"conversation body"]= str(c_item['body'])
        # if c_item['assigned_to'] is not None:
            # intercom_json[f"conversation {i} conversation part {j} assigned_to type"]= c_item['assigned_to']['type']
        if c_item['author'] is not None: 
            intercom_json[f"conversation author name"]= c_item['author']['name']
            # intercom_json[f"conversation {i} conversation part {j} author type"]= c_item['author']['type']
            # intercom_json[f"conversation {i} conversation part {j} author mail"]= c_item['author']['email']
        # if c_item['attachments'] !=[]: 
            # intercom_json[f"conversation {i} conversation part {j} attachments"]= c_item['attachments']
        intercom.append(intercom_json)

    json_object = json.dumps(intercom, indent=4)
    os.makedirs('./'+str(path),exist_ok=True)
    filename = './'+str(path)+"intercom.json"
    with open(filename, "a") as outfile:
        outfile.write(json_object)
    return intercom,filename

def clean_jira(data,path):
    jira_json={}

    jira_json["total issues"]= data['total']
    i=1
    for item in data["issues"]:
        if item['fields']['resolution'] is not None:
            jira_json[f"issue: {i} resolution description"]= item['fields']['resolution']['description']
            jira_json[f"issue: {i} resolution name"]=item['fields']['resolution']['name']

        jira_json[f"issue: {i} priority name"]= item['fields']['priority']['name']

        if item['fields']['labels']!=[]:
            j=1
            for l_items in item['fields']['labels']:
                jira_json[f"issue: {i} label {str(j)}"]= l_items
                j=j+1
        
        if item['fields']['issuelinks']!=[]:
            for is_item in item['fields']['issuelinks']:
                jira_json[f"issue: {i} type name"]= is_item['type']['name']
                try:
                    jira_json[f"issue: {i} inwardIssue summary"]= is_item['inwardIssue']['fields']['summary']
                    jira_json[f"issue: {i} inwardIssue status name"]= is_item['inwardIssue']['fields']['status']['name']
                    jira_json[f"issue: {i} inwardIssue status description"]= is_item['inwardIssue']['fields']['status']['description']
                    jira_json[f"issue: {i} inwardIssue status category"]= is_item['inwardIssue']['fields']['status']['statusCategory']['name']
                    jira_json[f"issue: {i} inwardIssue priority"]= is_item['inwardIssue']['fields']['priority']['name']
                    jira_json[f"issue: {i} inwardIssue issuetype name"]= is_item['inwardIssue']['fields']['issuetype']['name']
                    jira_json[f"issue: {i} inwardIssue issuetype description"]= is_item['inwardIssue']['fields']['issuetype']['description']
                except:
                    pass

        jira_json[f"issue: {i} assignee displayName"]= item['fields']['assignee']['displayName']
        jira_json[f"issue: {i} assignee active"]= str(item['fields']['assignee']['active'])

        jira_json[f"issue: {i} status name"]= item['fields']['status']['name']
        jira_json[f"issue: {i} status description"]= item['fields']['status']['description']

        jira_json[f"issue: {i} creator displayName"]= item['fields']['creator']['displayName']
        jira_json[f"issue: {i} creator active"]= str(item['fields']['creator']['active'])

        jira_json[f"issue: {i} reporter displayName"]= item['fields']['reporter']['displayName']
        jira_json[f"issue: {i} reporter active:"]= str(item['fields']['reporter']['active'])
        jira_json[f"issue: {i} issuetype name"]= item['fields']['issuetype']['name']
        jira_json[f"issue: {i} issuetype description"]= item['fields']['issuetype']['description']

        jira_json[f"issue: {i} created"]= item['fields']['created']

        # file.write(f"description:\n")
        text=""
        for ct_item in item['fields']["description"]["content"]:
            for cnt_item in ct_item["content"]:
                try:
                    if len(cnt_item['text'])>1:
                        text=text+cnt_item['text']+" "
                except:
                    pass
        jira_json[f"issue: {i} description text"]= text
        jira_json[f"issue: {i} summary"]= item['fields']['summary']
        jira_json[f"issue: {i} duedate"]= item['fields']['duedate']
        i=i+1

    # json_object = json.dumps(jira_json, indent=4)
    json_object = json.dumps(jira_json, indent=4)
    os.makedirs('./'+str(path),exist_ok=True)
    filepath = './'+str(path)+"jira.json"
    with open(filepath, "a") as outfile:
        outfile.write(json_object)
    return [jira_json], filepath

def clean_meet(data,path):
    sorted_json=[]
    participants={}

    for item in data["attendeeslist"]:
        participants[item["userId"]]=item["displayName"]

    metadata={}
    metadata["meetingtitle"]=data["meetingtitle"]
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