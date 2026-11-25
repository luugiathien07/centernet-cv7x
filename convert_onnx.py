import cflite

# Giả sử API chuyển đổi
cflite.convert('models/ctdet_coco_dla_2x_conv.onnx', 'model.cnngen')
print("Chuyển đổi ONNX sang CNNGen xong")