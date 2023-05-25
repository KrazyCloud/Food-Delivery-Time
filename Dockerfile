FROM python:latest
WORKDIR /usr/app/src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
CMD ["python", "app.py"]
