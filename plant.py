from flask import Flask, render_template, request
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import json
import base64

app = Flask(__name__)


model_url = "https://www.kaggle.com/models/rishitdagli/plant-disease/TensorFlow2/plant-disease/1"
model = hub.load(model_url)

with open('/home/mgarang/Downloads/class_indices.json', 'r') as f: 
    class_indices = json.load(f)
labels = [class_indices[str(i)] for i in range(len(class_indices))]

def preprocess_image(image):
    img = Image.fromarray(image)
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img)
    img_array = (img_array.astype(np.float32) / 255.0)  
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def predict_disease(image):
    processed_image = preprocess_image(image)

    logits = model(processed_image)
    predicted_class = np.argmax(logits)
    confidence = np.max(tf.nn.softmax(logits))  # Apply softmax for confidence
    disease_label = labels[predicted_class]
    return disease_label, confidence

treatment_data = {
    "healthy": "Your plant looks healthy! Keep giving it the right amount of water and sunshine.",
    "Corn___Common_rust": [
        "Ask a grown-up to help you find a fungicide at the garden store that's made for rust on corn. Look for words like **azoxystrobin**, **propiconazole**, or **tebuconazole** on the label.",
        "A grown-up should read the instructions on the fungicide very carefully and help you spray it on the corn plants.",
        "Make sure the air can move around the corn plants. Sometimes planting them with a bit of space between them helps.",
        "If your corn keeps getting rust, maybe next time you can try planting a type of corn that doesn't get rust easily."
    ],
    "Tomato___Early_blight": [
        "Pick off any leaves that have brown spots, especially the ones near the bottom of the plant. Put them in the trash so the spots don't spread.",
        "Ask a grown-up to get a copper spray from the garden store. This can help stop the spots from getting worse. They can help you spray it on the tomato plants.",
        "Try to water your tomato plants at the bottom, near the soil, so the leaves stay dry.",
        "Give your tomato plants some room to grow so the air can flow around them."
    ],
    "Potato___Late_blight": [
        "This is a tricky one, so get a grown-up's help quickly! If you see brown or black spots on the potato leaves or stems, those parts need to be taken off and thrown away (not in the compost).",
        "There are special sprays called fungicides that can help. A grown-up can find one for late blight on potatoes at the garden store and help you use it safely.",
        "Try not to get the potato leaves too wet when you water.",
        "Make sure the soil around your potatoes drains well and doesn't stay soggy."
    ],
    "Apple___Apple_scab": [
        "Ask a grown-up to help you find a special spray at the garden store for apple trees that get spots. Look for words like **captan** or **myclobutanil** on the label.",
        "The best time to spray is usually in the spring when the new leaves are coming out. A grown-up can help you figure out when to start.",
        "Make sure to spray all the leaves and even the little apples, following the instructions on the spray bottle with a grown-up's help.",
        "When the leaves fall off the tree in the autumn, ask a grown-up to help you rake them up and throw them away. These old leaves can have the stuff that causes the spots.",
        "Sometimes, planting a type of apple tree that doesn't get spots easily can help too!"
    ],
    "grape___Black_rot": [
        "If you see little white spots on the grape leaves that turn brown or black, or if the grapes look like they're rotting, ask a grown-up for help.",
        "A grown-up might need to get a special spray called a fungicide for grapes from the garden store.",
        "Make sure to spray all the grapevines, following the instructions on the bottle with a grown-up's help.",
        "It helps to keep the area around the grapevines clean. Ask a grown-up to help you pick up any old leaves or yucky grapes from the ground.",
        "Grapes like to have air moving around them, so make sure the vines aren't too crowded."
    ],
    "blueberry___healthy": "Your blueberry plant looks great! Keep giving it water when it's dry and make sure it gets plenty of sunshine.",
    "cherry___Powdery_mildew": [
        "If you see a white, powdery stuff on the cherry leaves, ask a grown-up for help.",
        "There are special sprays a grown-up can get from the garden store to help with this. Look for things like **neem oil** or **sulfur** on the label.",
        "Sometimes, just spraying the leaves with water in the morning can help wash off the white powder, but don't do it too much so the leaves don't stay wet all day.",
        "Cherry trees like to have air moving around their leaves, so a grown-up might need to trim some branches if it's too crowded."
    ],
}

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    prediction = None
    confidence = None
    treatment = None
    uploaded_image = None

    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            try:
                img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
                disease, conf = predict_disease(img)
                prediction = disease
                confidence = f"{conf * 100:.2f}%"  # Show confidence as a percentage
                treatment = treatment_data.get(disease, "Hmm, I don't have specific treatment info for this right now. You might want to ask a plant expert at a garden store!")
                uploaded_image = 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')
            except Exception as e:
                prediction = "Oops! Something went wrong with the prediction."
                print(f"Error: {e}")

    return render_template('index.html', prediction=prediction, confidence=confidence, treatment=treatment, uploaded_image=uploaded_image)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
