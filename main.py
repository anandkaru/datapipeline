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

bucket_path = 'agent-assist-embeddings/'
path_to_private_key = './saas-labs-staging-rnd-0affa0ea1703.json'
client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)

bucket = storage.Bucket(client, 'saas-labs-staging-rndaip-qs1h0h8x')

def upload_folder_to_bucket(source_folder, destination_folder):
  

    # Iterate through the files in the source folder
    for root, _, files in os.walk(source_folder):
        for file_name in files:
            # Create the full local path to the file
            local_file_path = os.path.join(root, file_name)

            # Create the corresponding blob name in the destination folder of the bucket
            blob_name = os.path.join(destination_folder, file_name)

            # Upload the file to the bucket
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file_path)

            print(f"Uploaded {local_file_path} to gs:/{blob_name}")


def download_folder_from_bucket( source_folder, destination_folder):
   

    # Get a list of blobs in the specified source folder
    blobs = bucket.list_blobs(prefix=source_folder)

    # Iterate through the blobs and download each file
    for blob in blobs:
        # Get the relative path of the file inside the source folder
        relative_path = os.path.relpath(blob.name, source_folder)

        # Create the local file path where the file will be downloaded
        local_file_path = os.path.join(destination_folder, relative_path)

        # Create the directories if they don't exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the file from the bucket to the local file path
        blob.download_to_filename(local_file_path)

        print(f"Downloaded gs://{blob.name} to {local_file_path}")


def clean_data_to_format(data,source):
    if source.lower()=='jira':
        cleaned_data=clean_jira(data)
    elif source.lower()=='intercom':
        cleaned_data=clean_intercom(data)
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

def save_embeddings(id):
    # DRIVE_FOLDER = "/content/drive/MyDrive/lang_data/Real_jsons"
    DRIVE_FOLDER = "/content/drive/MyDrive/lang_data"
    loader_json = DirectoryLoader(DRIVE_FOLDER, glob='**/*.json', show_progress=True, loader_cls=TextLoader)
    loaders = [loader_json]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = char_text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    folder_path = path + str(id) + "_faiss_index"
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
    cleaned_data = clean_data_to_format(data,source)
    #upload_json_to_bucket(json_path)
    ##wait_5s
    folder_path = save_embeddings(id)
    upload_folder_to_bucket(folder_path)

@app.post('/contextai/update')
async def update(request: Request):
    body = await request.json() 
    companyid = body['companyid']  
    source = body['source'] 
    sourceid = get_id(companyid,source)
    data = get_data(sourceid)
    cleaned_data = clean_data_to_format(data,source)
    #upload_json_to_bucket(json_path)
    ##wait_5s
    # folder_path = save_embeddings(id)
    download_folder_from_bucket(id,'./')
    merge(cleaned_data,'./'+id+'_faiss_index')
    upload_folder_to_bucket('./'+id+'_faiss_index')