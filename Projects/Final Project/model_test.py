import json
from time import sleep
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import os
import paho.mqtt.client as mqtt

def load_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_tflite_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

def on_disconnect(client, userdata, rc):
    print("Disconnected with result code " + str(rc))

while True:
    count_empt = 0
    count_occp = 0

    MQTT_HOST = '10.52.138.73'
    MQTT_PORT = 1883
    MQTT_TOPIC = 'parking_space'
    MQTT_CLIENT_ID = 'raspi'

    # 载入图片
    image = Image.open("./test.jpeg")

    parking_spots = [
        # (250, 1830, 1821, 2691),
        # (1032, 1710, 2397, 2295),
        # (1602, 1619, 2792, 2066),
        # (2020, 1546, 3092, 1905),
        # (2340, 1487, 3320, 1790)
        (520, 1944, 1492, 2438),
        (1386, 1766, 2043, 2227),
        (1930, 1641, 2484, 2025),
        (2237, 1649, 2771, 1753),
        (2581, 1523, 3125, 1701)
    ]

    for i, (left, top, right, bottom) in enumerate(parking_spots, start=1):
        cropped_image = image.crop((left, top, right, bottom))

        cropped_image.save(f"./test/parking_spot_{i}.jpg")
        print(f"Saved parking spot {i}.")



    model_path = './parking_lot_occupancy_model_logback.tflite'

    for file in os.listdir('./test'):
        file_path = os.path.join('./test', file)
        if os.path.isfile(file_path):
            input_data = load_image(file_path)
            interpreter = load_tflite_model(model_path)
            output_data = run_inference(interpreter, input_data)
            print("Predicted output:", output_data)

            if (output_data > 0.5):
                count_occp += 1
            else:
                count_empt += 1

    client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    try:
        client.connect(MQTT_HOST, MQTT_PORT, 60)
        client.loop_start()
        message = json.dumps([count_occp, count_empt])
        client.publish(MQTT_TOPIC, message)
    finally:
        client.loop_stop()
        client.disconnect()
    print("Message published")
    print("result is: ", message)
    sleep(5)