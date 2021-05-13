# docker image build -t streamlit:app .
# docker container run -p 8501:8501 -d streamlit:app

FROM ubuntu:20.04
RUN apt-get update && apt-get install -y \
    python3-pip
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 8501
CMD streamlit run app.py