import numpy as np
from google.cloud import storage
import tensorflow as tf
from PIL import Image

model = None
interpreter = None
input_index = None
output_index = None

class_names = ['Bercak Abu-Abu (Cercospora)', 'Hawar Daun', 'Karat Daun', 'Sehat']

BUCKET_NAME = "corn-disease-model"

def download_blob(bucket_name, source_blob_name, destination_file_name) :
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}")


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/modelv2.h5",
            "/tmp/modelv2.h5"
        )
        model = tf.keras.models.load_model("/tmp/modelv2.h5")


    image = request.files["file"]

    image = np.array(
        Image.open(image)
    )

    image = tf.image.resize(image,[256,256])

    img_array = tf.expand_dims(image,0)
    predictions = model.predict(img_array)

    print(f"Predictions : {predictions}")

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]),2)


    return {"class" : predicted_class, "confidence" : float(confidence)}