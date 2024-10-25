
"""
    Requires CompressAI environment

    Performs the inference from the exported ONNX version of the deep learning model
    
    Saves the compressed image to the disc at the provided path 
    
    Gives the time taken for the inference in both the RGB veersion and the 3 channel grayscale version
"""

import argparse
import cv2
import onnx
import numpy as np
import onnxruntime
from PIL import Image
import time
import io


def preprocess_gray(image_path):
    """
    Function to convert the image into the grayscale version of 3 channels
    Where 1 channel has the grayscale data and the other two channels have 0s
    """
    
    image=Image.open(image_path).convert("L") #convert the image to the Grayscale
    #resize the image
    image=image.resize((1640,1232))
    #convert to numpy and normalize
    image=np.array(image)/255.0
    
    #create an empty numpy array
    img_blank=np.zeros((3,1232,1640))

    #assign the gray image to the blank array
    img_blank[0]=image
    
    
    img_final=np.expand_dims(img_blank,axis=0).astype(np.float32)

   
    return img_final

def preprocess_simple_gray(image_path):

    """
    preprocesses the image into simple grayscale of 2 dimension
    """
    image=Image.open(image_path).convert("L")
    image=image.resize((1232,1640))
    image=np.array(image)/255.0
    image=image.transpose()
    image=np.expand_dims(image,axis=0).astype(np.float32)
    image=np.expand_dims(image,axis=0)
    return image


def preprocess_image(image_path):
    """
    Preprocess the input image to the format expected by the model.
    RGB version
    """
    # Load the image using PIL
    image = Image.open(image_path).convert("RGB")

    image=image.resize((1640,1232))
    
    # Convert to numpy array and normalize the image (0 to 1)
    image = np.array(image) / 255.0
    
    # Convert to CHW format (Channels x Height x Width)
    image = np.transpose(image, (2, 0, 1))
    
    # Add a batch dimension (1, Channels, Height, Width)
    image = np.expand_dims(image, axis=0).astype(np.float32)
    
    return image


def save_compressed_image(output, save_path):
    """
    Save the compressed output image from the model inference.
    """
    # Post-process the output (if needed)
    output = output.squeeze(0)  # Remove batch dimension
    
    # Convert to HWC format (Height x Width x Channels)
    output = np.transpose(output, (1, 2, 0))
    
    # Clip values to valid range (0, 1) and convert to 8-bit (0-255)
    output = np.clip(output, 0, 1) * 255.0
    output = output.astype(np.uint8)
    
    # Convert to image format and save
    output_image = Image.fromarray(output)
    output_image.save(save_path,format="WEBP")


def export_to_buffer(output):
    """
    Exports the image to the buffer instead of the disc
    For faster communication for the network.
    """

    #post process the output
    output=output.squeeze(0)

    #convert to the HWC format
    output=np.transpose(output,(1,2,0))

    #clip values to the valid range(0,1)
    output=np.clip(output,0,1)*255.0
    output=output.astype(np.uint8)

    #convert to PIL image and write the data to the in_memory buffer
    output_image=Image.fromarray(output)
    buffer_stream=io.BytesIO()
    output_image.save(buffer_stream,format="JPEG")

    return buffer_stream.getvalue()



def run_inference(image_path, model_path, save_path):
    """
    Run inference on an input image using the exported ONNX model.
    """
    # Load the image and preprocess it
    #input_image = preprocess_image(image_path)

    #gray image
    #input_image_gray=preprocess_gray(image_path)

    #simple 2 channel gray image
    input_image=preprocess_image(image_path)
    
    # Load the ONNX model

    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    #session_options.log_severity_level = onnxruntime.logging.LoggingLevel.WARNING

    #Check if CUDA is available
    providers = [('CUDAExecutionProvider',{"use_tf32":0})] if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']
    print("Using:",providers)
    onnx_session = onnxruntime.InferenceSession(model_path,sess_options=session_options, providers=providers)
    #onnx_session = onnxruntime.InferenceSession(model_path)

    
    # Run inference
    input_names = ["input"]
    output_names = ["output"]
    start=time.time()
    onnx_output = onnx_session.run(output_names, {input_names[0]: input_image})[0]
    end=time.time()

    print("Time in seconds for the gray image inference is:",round(end-start,2),"seconds")

    #run inference for the grayscale image
    #input_names=["input"]
    #output_names=["output"]
    #start=time.time()
    #onnx_output_gray=onnx_session.run(output_names,{input_names[0]:input_image_gray})[0]
    #end=time.time()

    #print("Time in seconds for the inference of gray is:",round(end-start,2),"seconds")
    

    
    # Save the output image
    start=time.time()
    save_compressed_image(onnx_output, save_path)
    end=time.time()
    print("Exporting to disc timing:",round(end-start,2),"seconds")

    start=time.time()
    buffer=export_to_buffer(onnx_output)
    end=time.time()
    print("Exporting to the buffere timing:",round(end-start,2),"seconds")
    print(f"Compressed image saved at: {save_path}")
    
    
def main(args):
    """Main function to run the inference from the onnx model"""
    run_inference(args.image_path, args.model_path, args.save_path)

if __name__=="__main__":
    
    #make the parser object
    parser=argparse.ArgumentParser(description="Script to perform inference on ONNX model on the dataset")
    
    
    parser.add_argument("--image_path",type=str,required=True,help="Path to the image file")
    parser.add_argument("--model_path",type=str,required=True,help="Path to the ONNX model file")
    parser.add_argument("--save_path",type=str,required=True,help="Path to save the output image")
    
    args=parser.parse_args()
    
    #run the inference from the model
    main(args)

