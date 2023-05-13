from flask import Flask, jsonify, render_template, request
import torch
import clip
from PIL import Image
import base64
import os
from torchvision.datasets import CIFAR100
import numpy as np


# Flask App
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def get_probs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    images = request.form['images']
    outputs = []
    for img in images:
        image = preprocess(Image.open(img)).unsqueeze(0).to(device)
        texts = request.form['texts']
        text = clip.tokenize(texts).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            output = f"-------------------- {img} -------------------------:"
            print(output)
            outputs.append(output)
            for i, text in enumerate(texts):
                output = f"{text}: {probs[0][i]}"
                print(output)
                outputs.append(output)
            outputs.append("")
    return jsonify(outputs)


# @app.route('/')
# def hello():
#     return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    # Prepare the inputs
    image, class_id = cifar100[3637]
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # Create a list of predictions
    predictions = []
    for value, index in zip(values, indices):
        prediction = {
            'class': cifar100.classes[index],
            'score': 100 * value.item()
        }
        predictions.append(prediction)

    # Return the predictions as a JSON response
    return jsonify(predictions)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run()
