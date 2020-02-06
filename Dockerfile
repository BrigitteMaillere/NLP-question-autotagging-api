FROM python:3.6

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

RUN [ "python", "-c", "import nltk; nltk.download('stopwords')" ]
RUN [ "python", "-c", "import nltk; nltk.download('punkt')" ]
RUN [ "python", "-c", "import nltk; nltk.download('wordnet')" ]

COPY . /app

ENTRYPOINT ["python"]

CMD ["app.py"]
