FROM python:3-slim-stretch
COPY ./src /src
WORKDIR /src
CMD [ "pyton wsgi.py" ]