import json 

def clean_intercom(data):
    intercom_json={}
    i=1
    for item in data:
        intercom_json[f"conversation {i} type"]= item['type']
        intercom_json[f"conversation {i} source delivered_as"]= item['source']['delivered_as']
        intercom_json[f"conversation {i} source subject"]= item['source']['subject']
        if item['source']['body'] is not None:
            intercom_json[f"conversation {i} source body"]= item['source']['body']
        if item['source']['author'] is not None:
            intercom_json[f"conversation {i} source author type"]= item['source']['author']['type']
            intercom_json[f"conversation {i} source author name"]= item['source']['author']['name']
            intercom_json[f"conversation {i} source author email"]= item['source']['author']['email']

        intercom_json[f"conversation {i} status"]= item['state']
        intercom_json[f"conversation {i} read"]= str(item['read'])
        intercom_json[f"conversation {i} priority"]= item['priority']
        if item['conversation_rating'] is not None:
            intercom_json[f"conversation {i} conversation rating"]= str(item['conversation_rating']['rating'])
            intercom_json[f"conversation {i} conversation remark"]= str(item['conversation_rating']['remark'])
            intercom_json[f"conversation {i} conversation contact type"]= item['conversation_rating']['contact']['type']
        j=1
        for c_item in item["conversation_parts"]["conversation_parts"]:
            intercom_json[f"conversation {i} conversation part {j} part_type"]= c_item['part_type']
            intercom_json[f"conversation {i} conversation part {j} body"]= str(c_item['body'])
            if c_item['assigned_to'] is not None:
                intercom_json[f"conversation {i} conversation part {j} assigned_to type"]= c_item['assigned_to']['type']
            if c_item['author'] is not None: 
                intercom_json[f"conversation {i} conversation part {j} author name"]= c_item['author']['name']
                intercom_json[f"conversation {i} conversation part {j} author type"]= c_item['author']['type']
                intercom_json[f"conversation {i} conversation part {j} author mail"]= c_item['author']['email']
            if c_item['attachments'] !=[]: 
                intercom_json[f"conversation {i} conversation part {j} attachments"]= c_item['attachments']
            j=j+1
        i=i+1
    json_object = json.dumps(intercom_json, indent=4)

    with open("intercom_single.json", "a") as outfile:
        outfile.write(json_object)
    return intercom_json

def clean_jira(data):
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

    with open("jira_json.json", "a") as outfile:
        outfile.write(json_object)
    return jira_json
