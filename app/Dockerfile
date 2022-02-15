FROM python:3.8-alpine

#RUN mkdir "app"
#WORKDIR /app

#RUN apk --update --upgrade add --no-cache  gcc musl-dev jpeg-dev zlib-dev libffi-dev cairo-dev pango-dev gdk-pixbuf-dev

RUN python -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


COPY . .


CMD [ "python", "app.py" ]

EXPOSE 5000