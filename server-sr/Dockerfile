FROM python:3

ADD main.py .

ADD model-en model

RUN pip install flask-restful vosk 

CMD ["python", "main.py"]
