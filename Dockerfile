FROM rayproject/ray:2.50.1-py39-gpu

RUN pip install torch --index-url https://download.pytorch.org/whl/cu126
RUN pip install swig
RUN pip install wandb[media] pillow gymnasium[box2d]

COPY . /app
WORKDIR /app