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
import json
import uvicorn
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

def download_folder_from_bucket(foldername, local_folder):
    blobs = bucket.list_blobs(prefix=foldername)
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, foldername)
        local_file_path = os.path.join(local_folder, relative_path)
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
    local_file_path=filename+"_duplicate"
    blob.download_to_filename(local_file_path)
    print(f"Downloaded gs://{blob.name} to {local_file_path}")
    return local_file_path
    
def clean_data_to_format(data,source,sourceid,i):
    if source.lower()=='jira':
        cleaned_data=clean_jira(data,sourceid,i)
    elif source.lower()=='intercom':
        cleaned_data=clean_intercom(data,sourceid,i)
    else:
        return {"message":"Source not found"}
    return cleaned_data

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

def save_embeddings(sourceid,localpath):
    # DRIVE_FOLDER = "/content/drive/MyDrive/lang_data/Real_jsons"
    # DRIVE_FOLDER = "/content/drive/MyDrive/lang_data"
    DRIVE_FOLDER =localpath
    loader_json = DirectoryLoader(DRIVE_FOLDER, glob='**/*.json', show_progress=True, loader_cls=TextLoader)
    loaders = [loader_json]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = char_text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    # flnm = filename.split(".")[0]
    folder_path = sourceid + "_faiss_index"
    vector_store.save_local(folder_path)
    print("Embeddings saved")
    shutil.rmtree(localpath)
    return folder_path

def merge(localpath,folderpath):
  db1 = FAISS.load_local(localpath, embeddings)
  db2 = FAISS.load_local(folderpath, embeddings)
  db1.merge_from(db2)
  try:
      shutil.rmtree(localpath)
      shutil.rmtree(folderpath)
  except OSError as e:
      print("Error: %s - %s." % (e.filename, e.strerror))
  db1.save_local(folderpath)
  return folderpath

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
    conversationid = body['conversationid']
    sourceid = str(get_id(companyid,source))
    data = get_data(sourceid)
    cleaned_data, filename = clean_data_to_format(data,source,sourceid,i=1)
    ##upload_data_to_mongo(cleaned_data)
    upload_file_to_bucket(filename)
    folder_path = save_embeddings(sourceid,'./jsons/')
    upload_folder_to_bucket(folder_path)

@app.post('/contextai/update')
async def update(request: Request):
    body = await request.json() 
    companyid = body['companyid']  
    source = body['source'] 
    conversationid = body['conversationid']
    sourceid = str(get_id(companyid,source))
    data = get_data(sourceid)
    duplicate_path = download_file_from_bucket(filename)
    f = open(duplicate_path)
    older_data = json.loads(f)
    f.close()
    i = older_data.keys()[-1].split(" ")[1]
    cleaned_data, filename = clean_data_to_format(data,source,sourceid,i+1)
    folderpath = save_embeddings(sourceid,'./jsons/')
    local_folder = sourceid+'_temp'
    bucket_folder = sourceid + "_faiss_index"
    download_folder_from_bucket(bucket_folder,local_folder)
    folderpath = merge(local_folder,folderpath)
    upload_folder_to_bucket(folderpath)
    merged_data = {**older_data, **cleaned_data}
    with open(filename, 'w') as file:
        json.dumps(merged_data, file, indent=4)
    os.remove(duplicate_path)
    ##upload_data_to_mongo(merged_data)
    upload_file_to_bucket(filename)


if __name__ == "__main__":
    uvicorn.run("main:app",host = '0.0.0.0', port = 8000)



from fastapi import FastAPI
from fastapi.logger import logger
from pydantic import BaseSettings
import sys
from pyngrok import ngrok

class Settings(BaseSettings):
    # ... The rest of our FastAPI settings

    BASE_URL = "http://localhost:8000"
    USE_NGROK = os.environ.get("USE_NGROK", "False") == "True"


settings = Settings()


def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    pass


# Initialize the FastAPI app for a simple web server
app = FastAPI()

if settings.USE_NGROK:
    # pyngrok should only ever be installed or initialized in a dev environment when this flag is set

    # Get the dev server port (defaults to 8000 for Uvicorn, can be overridden with `--port`
    # when starting the server
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 8000

    # Open a ngrok tunnel to the dev server
    public_url = ngrok.connect(port).public_url
    logger.info("ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

    # Update any base URLs or webhooks to use the public ngrok URL
    settings.BASE_URL = public_url
    init_webhooks(public_url)