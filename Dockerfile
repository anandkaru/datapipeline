FROM --platform=linux/amd64 nvidia/cuda:12.2.0-runtime-ubuntu20.04

WORKDIR /app


RUN apt update \
    && apt install -y python3-pip

COPY ./req.txt requirements.txt

RUN pip install -U sentence-transformers

RUN pip install google-api-python-client

RUN pip install --upgrade google-cloud-storage

RUN pip install pymongo[srv]

RUN pip install  -r requirements.txt

COPY . ./

EXPOSE 8000

CMD ["python3","main.py"]