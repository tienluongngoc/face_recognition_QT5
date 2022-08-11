python weights/remove_initializer.py --input weights/arcface/1/model.onnx --output weights/arcface/1/model_rminit.onnx
python weights/remove_initializer.py --input weights/scrfd/1/model.onnx --output weights/scrfd/1/model_rminit.onnx
python weights/remove_initializer.py --input weights/fasnet/v1se/model.onnx --output weights/fasnet/v1se/model_rminit.onnx
python weights/remove_initializer.py --input weights/fasnet/v2/model.onnx --output weights/fasnet/v2/model_rminit.onnx