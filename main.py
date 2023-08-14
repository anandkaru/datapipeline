import pymongo
import os
import shutil

from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

import logging
import json
from decouple import config

# from bson.objectid import ObjectId

from data_clean import clean_intercom, clean_jira, clean_meet, clean_zoom


os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')

client_url = config('CONNECTION_URL')
client = pymongo.MongoClient(client_url)
print("client ready")

db = client['cai']
# collection1 = db['ids']
collection2 = db['intercom_data']
target_data_collection = db['embedding_jsons']

# def get_id(companyid,source):
#     query = {"companyid": companyid}
#     result = collection1.find_one(query)
#     if result:
#         sourceid = result[source]
#         return sourceid
#     else:
#         print("Document not found.")

def get_data(companyid,customerid):
    query = {'companyid':companyid,'customerid':customerid}
    result = collection2.find(query)
    data_final=[]
    for ele in result:
        data = ele['data']
        data_final.append(data)
    path = companyid+'/'+customerid+'/'
    return data_final,path

def mongo_format(foldername):
    file_list = [f for f in os.listdir(foldername) if f.endswith('.json')]
    data_list=[]
    for filename in file_list:
        filepath = os.path.join(foldername, filename)
        
        with open(filepath, 'r') as f:
            json_data = f.read()
        json_data = json.loads(json_data)
            
        data_list.append({
            "source": filename[:-5],  # Remove the '.json' extension
            "data": json_data
        })

    return data_list

def mongo_update(companyid, customerid, data):
    complete_data = {
        "companyid": companyid,
        "customerid": customerid,
        "data": data
    }
    query = {'companyid':companyid,'customerid':customerid}
    existing_data = target_data_collection.find_one(query)
    if existing_data:
        target_data_collection.replace_one(query, complete_data)
    else:
        target_data_collection.insert_one(complete_data)

from google.cloud import storage

bucket_path_embeddings = 'cai-embeddings/'
bucket_path_jsons = 'cai-jsons/'
path_to_private_key = './creds.json'
client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)

bucket = storage.Bucket(client, 'saas-labs-staging-rndaip-qs1h0h8x')

def upload_folder_to_bucket(foldername,bucket_path):
    for root, _, files in os.walk(foldername):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            blob_name = os.path.join(bucket_path+foldername, file_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs:/{blob_name}")
    delete = foldername.split('/')
    shutil.rmtree(delete[0])
    print(f"Removed local folder: {foldername}")

def upload_file_to_bucket(filepath,bucket_path):
    path = bucket_path+filepath[2:]
    blob = bucket.blob(path)
    blob.upload_from_filename(filepath)
    os.remove(filepath)
    print(f"Uploaded local file: {filepath}")

def download_jsons_folder_from_bucket(foldername, bucket_path):
  blobs = bucket.list_blobs(prefix=bucket_path+foldername)
  for blob in blobs:
      relative_path = os.path.relpath(blob.name, foldername)
      arr = relative_path.split("/")
      # path = arr[-2]
      relative_path = '/'.join(arr[3:])
      print(relative_path)
      os.makedirs(foldername, exist_ok=True)
      blob.download_to_filename(relative_path)
      print(f"Downloaded gs://{blob.name} to {relative_path}")

def download_embeddings_folder_from_bucket(foldername,bucket_path):
  blobs = bucket.list_blobs(prefix=bucket_path+foldername)
  for blob in blobs:
      relative_path = os.path.relpath(blob.name, foldername)
      arr = relative_path.split("/")
      make_path = arr[-2]
      relative_path = '/'.join(arr[-2:])
      print(relative_path)
      os.makedirs(make_path, exist_ok=True)
      blob.download_to_filename(relative_path)
      print(f"Downloaded gs://{blob.name} to {relative_path}")
    
def clean_data_to_format(data,source,path):
    if source.lower()=='jira':
        cleaned_data, filename = clean_jira(data,path)
    elif source.lower()=='intercom':
        cleaned_data, filename = clean_intercom(data,path)
    elif source.lower()=='meet':
        cleaned_data, filename = clean_meet(data,path)
    elif source.lower()=='zoom':
        cleaned_data, filename = clean_zoom(data,path)
    else:
        return {"message":"Source not found"}
    return cleaned_data, filename

model_name = "sentence-transformers/all-MiniLM-L6-v2"
# model_kwargs = {'device': 'cuda'}
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    cache_folder='./embeddings',
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def save_embeddings(companyid,customerid):
    DRIVE_FOLDER ='./'+companyid+'/'+customerid+'/'
    loader_json = DirectoryLoader(DRIVE_FOLDER, glob='**/*.json', show_progress=True, loader_cls=TextLoader)
    loaders = [loader_json]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = char_text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    shutil.rmtree(DRIVE_FOLDER)
    temp = DRIVE_FOLDER+companyid+'_'+customerid+'_faiss_index/'
    vector_store.save_local(temp)
    print("Embeddings saved")
    return temp[2:]

def query_data(companyid,customerid,source):
  query = {'companyid':companyid,'customerid':customerid}
  existing_data = target_data_collection.find_one(query)
  data = existing_data['data']
  for i in range(len(data)):
    if data[i]['source']==source.lower():
      j_data = data[i]['data']
      return j_data

'''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

chain_collection={}
vector_store_collection={}

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0, max_tokens=512, verbose=True)#, device="cuda")  # Modify model_name if you have access to GPT-4


def initialize(companyid, customerid):
  id = companyid+'_'+customerid

  template="""You are a CEO of SaaS company and also a health score predictor of SaaS customers.
        As a CEO provide concise and on point answers relating to asked question.
        Health score methodology: Study complete data and then apply this logic, score will be very low(from 0 to 3) if majority of high priority important issues are not resolved, score will be moderate(from 4 to 7) if we are able to resolve important issues but not all issues and score will be on higher side(from 8 to 10) only if we have resolved most of the issues including all priority issues.
        At last give a specific number for health score in the format of - The health score of the customer would be i . Use the following pieces of context to answer the users question.
        Before giving the answer go through all the provided data such that there is information in the memory from all sorts of data.
        If you don't know the answer, don't try to make up an answer.

        {context}

        {chat_history}
        Human: {human_input}
        Chatbot:
        """


  prompt = PromptTemplate(input_variables=["chat_history", "human_input", "context"], template=template)

  memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
  foldername = companyid+'/'+customerid+'/'+companyid+'_'+customerid+'_faiss_index'
  download_embeddings_folder_from_bucket(foldername,bucket_path_embeddings)
  filepath=companyid+'_'+customerid+'_faiss_index'
  vector_store = FAISS.load_local(filepath, embeddings)
  vector_store_collection[id]=vector_store
  shutil.rmtree(filepath)
  chain = load_qa_chain(
      llm=llm,
      chain_type="stuff",
      memory=memory,
      prompt=prompt)

  print(" chain initialized: " + str(id))
  chain_collection[str(id)] = chain
  print(chain_collection.keys())


def output(query, companyid, customerid):
  id = companyid+'_'+customerid
  logging.getLogger("openai").setLevel(logging.DEBUG)
  chain =  chain_collection[str(id)]
  vector_store=vector_store_collection[id]
  doc=vector_store.similarity_search(query)

  with get_openai_callback() as cb:
    result=chain({"input_documents": doc, "human_input": query}, return_only_outputs=False)
    tokens_used=cb.total_tokens

  return [query , result['output_text'], tokens_used]

def qa(query, companyid,customerid):
    answer = output(query, companyid, customerid)
    print(f"query:{answer[0]}")
    print(f"Answer:{answer[1]}")
    print(f"Tokens used:{answer[2]}")
    data = {"query":answer[0],"answer":answer[1],"tokens_used":answer[2]}
    return data

from fastapi import FastAPI, Request
import uvicorn

app = FastAPI() 

@app.get('/contextai/health')
async def health():
    return {"message": 'healthy'}

@app.post('/contextai/embeddings')
async def saving_data(request: Request):
    body = await request.json()  
    companyid = body['companyid']  
    customerid = body['customerid']
    source = body['source'] 
    data, path = get_data(companyid,customerid)
    _, filename = clean_data_to_format(data,source,path)
    upload_file_to_bucket(filename,bucket_path_jsons)
    foldername=companyid+'/'+customerid+'/'
    download_jsons_folder_from_bucket(foldername,bucket_path_jsons)
    format_data = mongo_format(foldername)
    mongo_update(companyid,customerid,format_data)
    embeddings_path = save_embeddings(companyid,customerid)
    upload_folder_to_bucket(embeddings_path,bucket_path_embeddings)
    return {"message":"Data Successfully uploaded and Embeddings Saved Successfully for companyid: {} and customerid: {}".format(companyid,customerid)}
    # return {"message":"Data Successfully uploaded of source: {} for companyid: {} and customerid: {}".format(source,companyid,customerid)}

# @app.post('/contextai/embeddings')
# async def saving_embeddings(request: Request):
#     body = await request.json()  
#     companyid = body['companyid']  
#     customerid = body['customerid']
#     foldername=companyid+'/'+customerid+'/'
#     download_jsons_folder_from_bucket(foldername,bucket_path_jsons)
#     format_data = mongo_format(foldername)
#     mongo_update(companyid,customerid,format_data)
#     embeddings_path = save_embeddings(companyid,customerid)
#     upload_folder_to_bucket(embeddings_path,bucket_path_embeddings)
#     return {"message":"Embeddings Saved Successfully for companyid: {} and customerid: {}".format(companyid,customerid)}

@app.post('/contextai/initialize')
async def initalizing_embeddings(request: Request):
    body = await request.json() 
    companyid = body['companyid']  
    customerid = body['customerid']
    initialize(companyid, customerid)
    return {"message":"Initialization Successful for companyid: {} and customerid: {}".format(companyid,customerid)}

@app.post('/contextai/query')
async def query_answers(request: Request):
    body = await request.json() 
    companyid = body['companyid']  
    customerid = body['customerid']
    query = body['query']
    answer = qa(query, companyid,customerid)
    return answer

if __name__ == "__main__":
    uvicorn.run("main:app",host = '0.0.0.0', port = 8000)
