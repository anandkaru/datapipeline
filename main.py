import pymongo
import os
import io
import numpy as np
from decouple import config
import numpy as np
import shutil
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader, DirectoryLoader, TextLoader
import time
from data_clean import clean_jira, clean_intercom

client_url = config('CONNECTION_URL')
client = pymongo.MongoClient(client_url)
print("client ready")

db = client['aimodels']
collection1 = db['ids']
collection2 = db['data']

def get_id(companyid,source):
    query = {"companyid": companyid}
    result = collection1.find_one(query)
    if result:
        sourceid = result[source]
        return sourceid
    else:
        print("Document not found.")

def get_data(sourceid):
    query = {"_id": sourceid}
    result = collection2.find_one(query)
    return result

from google.cloud import storage

bucket_path = 'cai-embeddings/'
path_to_private_key = './saas-labs-staging-rnd-0affa0ea1703.json'
client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)

bucket = storage.Bucket(client, 'saas-labs-staging-rndaip-qs1h0h8x')

def upload_folder_to_bucket(foldername):
    for root, _, files in os.walk(foldername):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            blob_name = os.path.join(foldername, file_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs:/{blob_name}")
    shutil.rmtree(foldername)
    print(f"Removed local folder: {foldername}")

def download_folder_from_bucket(foldername):
    blobs = bucket.list_blobs(prefix=foldername)
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, foldername)
        local_file_path = os.path.join(foldername, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
        print(f"Downloaded gs://{blob.name} to {local_file_path}")

def upload_file_to_bucket(filename):
    path = bucket_path+filename
    blob = bucket.blob(path)
    blob.upload_from_filename(filename)
    os.remove(filename)

def download_file_from_bucket(filename):
    blob = bucket.blob(filename)
    local_file_path=filename+"(duplicate)"
    blob.download_to_filename(local_file_path)
    print(f"Downloaded gs://{blob.name} to {local_file_path}")
    
def clean_data_to_format(data,source,sourceid):
    if source.lower()=='jira':
        cleaned_data=clean_jira(data,sourceid)
    elif source.lower()=='intercom':
        cleaned_data=clean_intercom(data,sourceid)
    else:
        return {"message":"Source not found"}
    return cleaned_data


path = "./store/"

chain_collection={}

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}


embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    cache_folder=path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def save_embeddings(filename):
    # DRIVE_FOLDER = "/content/drive/MyDrive/lang_data/Real_jsons"
    # DRIVE_FOLDER = "/content/drive/MyDrive/lang_data"
    DRIVE_FOLDER = bucket_path+filename
    loader_json = DirectoryLoader(DRIVE_FOLDER, glob='**/*.json', show_progress=True, loader_cls=TextLoader)
    loaders = [loader_json]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = char_text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    flnm = filename.split(".")[0]
    folder_path = path + flnm + "_faiss_index"
    vector_store.save_local(folder_path)
    print("Embeddings saved")
    return folder_path

def merge(data,path):
  db1 = FAISS.load_local(path + str(id) + "_faiss_index", embeddings)
  db2 = FAISS.from_documents(data, embeddings)
  db1.merge_from(db2)
  try:
      shutil.rmtree(path + str(id) + "_faiss_index")
  except OSError as e:
      print("Error: %s - %s." % (e.filename, e.strerror))
  db1.save_local(path + str(id) + "_faiss_index")
  return 

from fastapi import FastAPI, Request

app = FastAPI() 

@app.get('/contextai/health')
async def health():
    return {"message": 'healthy'}

@app.post('/contextai/embeddings')
async def saving(request: Request):
    body = await request.json()  
    companyid = body['companyid']  
    source = body['source']  
    sourceid = get_id(companyid,source)
    data = get_data(sourceid)
    cleaned_data, filename = clean_data_to_format(data,source)
    upload_file_to_bucket(filename)
    ##upload_data_to_mongo(cleaned_data)
    time.sleep(5)
    folder_path = save_embeddings(filename)
    upload_folder_to_bucket(folder_path)

@app.post('/contextai/update')
async def update(request: Request):
    body = await request.json() 
    companyid = body['companyid']  
    source = body['source'] 
    sourceid = get_id(companyid,source)
    data = get_data(sourceid)
    cleaned_data, filename = clean_data_to_format(data,source,sourceid)
    download_file_from_bucket(filename)
    ##merge_data
    upload_file_to_bucket(filename)###merge
    ##wait_5s
    # folder_path = save_embeddings(id)
    download_folder_from_bucket(id,'./')
    merge(cleaned_data,'./'+id+'_faiss_index')
    upload_folder_to_bucket('./'+id+'_faiss_index')