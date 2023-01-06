FROM tensorflow/tensorflow:latest
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000:5000
ENTRYPOINT python app.py 5000
# CMD curl -X POST -F image=@img_12.jpg http://127.0.0.1:5000/