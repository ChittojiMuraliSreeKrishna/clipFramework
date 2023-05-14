from flask import Flask, jsonify, render_template, request
import torch
import clip
from PIL import Image
import base64
import os
from torchvision.datasets import CIFAR100
import numpy as np
from io import BytesIO

# Flask App
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def get_probs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outputs = []
    values = []
    if request.method == 'POST':
        model, preprocess = clip.load("ViT-B/32", device=device)
        images = request.files.getlist('images[]')
        texts = request.form.getlist('texts[]')
        for text in texts:
            new_values = text.split(',')
            for value in new_values:
                value = value.strip()
                values.append(value)
            print(values)
        for img in images:
            # Use BytesIO to create a file-like object from the uploaded image
            img_bytes = img.read()
            img_file = BytesIO(img_bytes)
            # Open the image using PIL's Image module
            image = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
            text = clip.tokenize(values).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                output = f"image-name: {img.filename} "
                print(output)
                outputs.append(output)
                output = f"---------- probabilities ----------"
                outputs.append(output)
                for i, text in enumerate(values):
                    output = f"{text} :- {probs[0][i]}"
                    print(output)
                    outputs.append(output)
                outputs.append("")
    return render_template('clip.html', outputs=outputs)


#     return jsonify(outputs)
# Run the app
if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run()
