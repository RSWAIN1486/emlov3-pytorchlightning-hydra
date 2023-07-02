FROM python:3.10-slim-buster

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt && rm -rf /root/.cache/pip

COPY . .

RUN pip install -e .



# CMD [ "python3", "src/train.py" ]
# CMD ["sh", "-c", "python3 src/train.py && python3 src/eval.py"]