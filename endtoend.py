import pymongo
import os
import io
import numpy as np
import numpy as np
import shutil
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader, DirectoryLoader, TextLoader
import time
import json
from bson.objectid import ObjectId
from data_clean import clean_jira, clean_intercom, clean_meet

client_url = 'mongodb+srv://teamuser:JIDPtRECf1qeAlNT@cluster0.suzik.mongodb.net'
client = pymongo.MongoClient(client_url)
print("client ready")

db = client['cai']
collection1 = db['ids']
collection2 = db['intercom_data']

def get_id(companyid,source):
    query = {"companyid": companyid}
    result = collection1.find_one(query)
    if result:
        sourceid = result[source]
        return sourceid
    else:
        print("Document not found.")

def get_data(companyid,customerid):
    query = {'companyid':companyid,'customerid':customerid}
    result = collection2.find(query)
    data_final=[]
    for ele in result:
        data = ele['data']
        data_final.append(data)
    path = companyid+'/'+customerid+'/'
    return data_final,path

from google.cloud import storage

bucket_path_embeddings = 'cai-embeddings/'
bucket_path_jsons = 'cai-jsons/'
path_to_private_key = './saas-labs-staging-rnd-0affa0ea1703.json'
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
    else:
        return {"message":"Source not found"}
    return cleaned_data, filename

# path = "./store/"

chain_collection={}

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}
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
    temp = DRIVE_FOLDER+companyid+customerid+'_faiss_index/'
    vector_store.save_local(temp)
    print("Embeddings saved")
    return temp[2:]


from fastapi import FastAPI, Request

app = FastAPI() 

@app.get('/contextai/health')
async def health():
    return {"message": 'healthy'}

@app.post('/contextai/datacollection')
async def saving(request: Request):
    body = await request.json()  
    companyid = body['companyid']  
    customerid = body['customerid']
    source = body['source'] 
    data, path = get_data(companyid,customerid)
    cleaned_data, filename = clean_data_to_format(data,source,path)
    upload_file_to_bucket(filename,bucket_path_jsons)
    return {"message":"Data Successfully uploaded of source: {} for companyid: {} and customerid: {}".format(source,companyid,customerid)}

@app.post('/contextai/embeddings')
async def saving(request: Request):
    body = await request.json()  
    companyid = body['companyid']  
    customerid = body['customerid']
    foldername=companyid+'/'+customerid+'/'
    download_jsons_folder_from_bucket(foldername,bucket_path_jsons)
    embeddings_path = save_embeddings(companyid,customerid)
    upload_folder_to_bucket(embeddings_path,bucket_path_embeddings)
    return {"message":"Embeddings Saved Successfully for companyid: {} and customerid: {}".format(companyid,customerid)}

@app.post('/contextai/intialize')
async def download(request: Request):
    body = await request.json() 
    companyid = body['companyid']  
    customerid = body['customerid']
    foldername = companyid+'/'+customerid+'/'+companyid+customerid+'_faiss_index'
    download_embeddings_folder_from_bucket(foldername,bucket_path_embeddings)
    return {"message":"Initialization Successful for companyid: {} and customerid: {}".format(companyid,customerid)}
