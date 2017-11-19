FROM python:3.6

ADD . /app

ENV PYTHONPATH /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["web/boot.py"]

