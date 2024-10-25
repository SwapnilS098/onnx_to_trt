

Here we are trying to make a complete pipeline from 

fine tune model (.pth version) -> loading model -> convert to gray ->
onnx version -> tensorRT version -> TRT inference

then do the result evaluation.


-Sample command to run the model_export_pytorch_onnx.py script

python model_export_pytorch_onnx.py --model_name bmshj2018-factorized --quality 4 --metric mse --channel 3 --h_w_shape 1232 1640 --export_path C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\fine_tune_to_trt --export_name bmshj_4_onnx.onnx


-sample command 
python model_export_pytorch_onnx.py --channel 3 --h_w_shape 1232 1640 --image_path C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\fine_tune_to_trt\image.png --quality 4 --export_name bmshj_4_fine_tune.onnx --model_path C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\fine_tune_to_trt\checkpoint.pth.tar