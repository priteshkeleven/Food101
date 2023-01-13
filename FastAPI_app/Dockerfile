FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
WORKDIR /code
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pillow
RUN pip install tensorflow==2.9.1

COPY ./main.py /code/main.py
COPY ./model /code/model

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]