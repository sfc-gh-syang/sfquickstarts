# NVIDIA container image for PyTorch
FROM nvcr.io/nvidia/pytorch:23.07-py3

WORKDIR /app

COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

ADD ./ /app

# Expose streamlit port
EXPOSE 8501
CMD ["streamlit", "run", "Home.py"]