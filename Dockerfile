# base image
FROM python:3.9

# making directory of app
WORKDIR mars_chatbot

# copy of requirements file
COPY requirements.txt ./requirements.txt

# install pacakges
RUN pip install -r requirements.txt

# copying all files over
COPY . .

# exposing default port for streamlit
EXPOSE 8501

# command to launch app
CMD streamlit run integrated_agents.py --client.showErrorDetails=False --theme.primaryColor=blue