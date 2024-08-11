import io
import base64
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

class MobileNetV3Handler(BaseHandler):
    def __init__(self):
        super(MobileNetV3Handler, self).__init__()
        self.model = None
        self.device = torch.device("cpu")
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest
        model_dir = context.system_properties.get("model-and-labels")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = 'model_epochs_10_IMAGENET1K_V2.pth'

        self.model = torch.load(model_pt_path, map_location=self.device)
        self.model.eval()

        self.class_names = ['occupied', 'unoccupied']  # Update this list based on your actual class names or load from a file

        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, data):
        images = []
        image_names = []

        # Access the first element in the list (assuming single request)
        request_data = data[0]

        # Access the 'cropped_images' key within the 'body' of the request_data
        cropped_images = request_data['body']['cropped_images']

        for image_data in cropped_images:
            image_name = image_data['name']
            image_base64 = image_data['image']
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image = self.data_transforms(image)
            images.append(image)
            image_names.append(image_name)

        return torch.stack(images).to(self.device), image_names

    def inference(self, x):
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities

    def postprocess(self, inference_output, image_names):
        results = []

        for index, (probability, image_name) in enumerate(zip(inference_output, image_names)):
            # Check if the output tensor is 1D or 2D
            if probability.dim() == 1:
                predicted_idx = torch.argmax(probability)  # Get the index of the max value in a 1D tensor
                confidence = torch.max(probability).item()
            else:
                _, predicted_idx = torch.max(probability, 1)  # Keep original behavior for 2D
                confidence = torch.max(probability, 1)[0].item()

            class_name = self.class_names[predicted_idx]
            
            # Adjust the result format as per the request
            result = {
                'index': index,
                'image_name': image_name,
                'prediction': {
                    'class': class_name,
                    'confidence': confidence
                }
            }
            
            results.append(result)

        # Return a single list of results
        return [results]

    def handle(self, data, context):
        self.initialize(context)
        images, image_names = self.preprocess(data)
        inference_output = self.inference(images)
        results = self.postprocess(inference_output, image_names)
        return results
