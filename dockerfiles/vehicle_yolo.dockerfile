FROM python:3.8

WORKDIR /project

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1 libglib2.0-0 && \
#     rm -rf /var/lib/apt/lists/*

COPY vehicle_yolo/requirements.txt ./
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

RUN pip uninstall opencv-python && \
    pip install opencv-python-headless
    
COPY vehicle_yolo/ ./

CMD ["python", "main.py"]


