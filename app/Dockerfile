FROM jupyter/tensorflow-notebook

#RUN mkdir "/app"
#USER root
#RUN mkdir -p "/app/uploads"
#RUN chmod -R 777 "/app/uploads"
#USER jovyan
WORKDIR /app
COPY . .


#RUN python -m pip install --upgrade pip setuptools wheel
#RUN python -m pip install --upgrade pip setuptools wheel
#COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


CMD [ "python", "app.py" ]

EXPOSE 5000