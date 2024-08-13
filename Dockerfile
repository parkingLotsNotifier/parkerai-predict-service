# Use the official TorchServe base image
FROM pytorch/torchserve:latest

# Switch to root user to install net-tools
USER root

# Install net-tools for netstat
RUN apt-get update && apt-get install -y net-tools

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the model file, handler, and other necessary files are in the appropriate directories
COPY model-and-labels /app/model-and-labels
COPY mobileNetV3Handler.py /app
COPY config.properties /app/config.properties

# Create the export_models directory
RUN mkdir -p ./export_models

# Archive the model
RUN torch-model-archiver --model-name mobilenet_v3 --version 1.0 --model-file mobileNetV3Handler.py --serialized-file ./model-and-labels/model_epochs_10_IMAGENET1K_V2.pth --handler mobileNetV3Handler --export-path ./export_models

# Expose the ports
EXPOSE 8080
EXPOSE 8081
EXPOSE 8082

# Command to start the service
CMD ["torchserve", "--start", "--ncs", "--model-store", "./export_models", "--models", "mobilenet_v3.mar", "--ts-config", "./config.properties"]

