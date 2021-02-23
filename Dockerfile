#FROM frolvlad:alpine-miniconda3
FROM continuumio/miniconda3

COPY environment.yml .

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "noe", "/bin/bash", "-c"]


ARG conda_env=noe

ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $conda_env


# RUN git clone https://github.com/rinikerlab/mlddec.git
RUN apt-get install unzip
RUN wget https://github.com/rinikerlab/mlddec/archive/master.zip && unzip master.zip && rm master.zip
WORKDIR mlddec-master 
RUN rm -r mlddec/data/epsilon_78
#can delete package after install?
RUN pip install . 


RUN echo 'conda activate noe' >> ~/.bashrc

##################
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=13579", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

#XXX need to switch to different branch??
COPY . /app/

WORKDIR /app/

RUN git checkout api

RUN pip install -e .

WORKDIR /app/examples/