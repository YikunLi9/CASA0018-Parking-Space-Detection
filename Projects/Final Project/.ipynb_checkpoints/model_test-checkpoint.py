import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import os

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

count_empt = 0
count_occp = 0

# 载入图片
image = Image.open("./IMG_2106.jpeg")  # 修改为你的图片文件路径

# 定义每个车位的边界，这些坐标需要根据实际图片进行调整
# 假设车位的坐标列表如下，每个元组表示一个车位的(left, top, right, bottom)
parking_spots = [
    # (250, 1830, 1821, 2691),   # 车位1
    # (1032, 1710, 2397, 2295),  # 车位2
    # (1602, 1619, 2792, 2066),   # 车位3
    # (2020, 1546, 3092, 1905),   # 车位4
    # (2340, 1487, 3320, 1790)
    (520, 1944, 1492, 2438),
    (1386, 1766, 2043, 2227),
    (1930, 1641, 2484, 2025),
    (2237, 1649, 2771, 1753),
    (2581, 1523, 3125, 1701)
]

# 切割并保存每个车位的图片
for i, (left, top, right, bottom) in enumerate(parking_spots, start=1):
    # 根据边界切割图片
    cropped_image = image.crop((left, top, right, bottom))

    # 保存切割后的图片，文件名对应车位编号
    cropped_image.save(f"./test/parking_spot_{i}.jpg")

    # 可选：显示切割后的图片
    cropped_image.show()


model_path = './parking_lot_occupancy_model_logback.tflite'

for file in os.listdir('./test'):
    input_data = load_image(file)
    interpreter = load_tflite_model(model_path)
    output_data = run_inference(interpreter, input_data)
    print("Predicted output:", output_data)

    if(output_data > 0.5):
        count_occp += 1
    else:
        count_empt += 1
