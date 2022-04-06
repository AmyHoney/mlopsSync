From paddlepaddle/paddle:2.2.2-jupyter

ENV NB_PREFIX /

RUN pip install --no-cache-dir \
notebook

EXPOSE 8888

CMD ["sh","-c", "jupyter notebook --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]

WORKDIR "/home/jovyan"
