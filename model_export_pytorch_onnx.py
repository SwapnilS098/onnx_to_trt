
"""
    Requires CompressAI environment
    
    -loads the model from the disc
    -gets the model from the compressai library
    -exports the model to onnx
    -preprocess the image
    -postprocess the image
    -runs the inference from the model 
    -converts the model to the grayscale version (single channel)
    -gets the model information   `                               
"""

#importing the modules
import argparse
import math
import io
import os
import torch

import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import time
import onnx
import onnxruntime

from onnx import numpy_helper
from PIL import Image

try:
    from compressai.zoo import models
except:
    print("compressAI env not activated")
    pass

import warnings
warnings.filterwarnings("ignore",category=torch.jit.TracerWarning)

#checking the device available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device="cpu"
print(device)


class ONNX_version:
    def __init__(self,args):
        
        if args.model_name:
            self.model_name=args.model_name
        else:
            self.model_name="bmshj2018-factorized"
            
        if args.quality==None:
            self.quality=4
        else:
            self.quality=args.quality
            
        if args.metric==None:
            self.metric="mse"
        else:
            self.metric=args.metric
            
            
        self.h_w_shape=args.h_w_shape
        
        if args.image_path==None:
            self.image_path=None 
        else:
            self.image_path=args.image_path
        
        self.export_path=args.export_path
        self.export_name=args.export_name
        
        
        self.batch_size=1
        
        self.model_path=args.model_path
        
        if args.second_quality:
            self.second_quality=args.second_quality
        else:
            self.second_quality=50
        
        if args.gray:
            self.gray=True
        else:
            self.gray=False
            
        if args.inference and args.image_path:
            print("\n image_path is given hence the inference will be run ")
            self.inference=True
        else:
            print("\n No image_path given hence no inference")
            self.inference=False
        
    def get_the_model(self):
        """
        this function get the model from the compressai
        models repo
        """
        
        
        #check if the model name is valid
        if self.model_name in models:
            print("Model exists")
        else:
            print("model_name:",self.model_name," is not a valid name")
            print()
            print("Models are:",list(models.keys()))
            print()
        model=models[self.model_name](quality=self.quality,metric=self.metric,pretrained=True)
        #print("PyTorch Model is:")
        #print(model)
        #print("======================================")
        #set the model to the device
        model.to(device)

        return model

    def load_model(self):
        """
        loads the model in the checkpoint.pth.tar format and 
        returns the model
        """
        
        #load the model
        model=self.get_the_model()
        
        #load the checkpoint
        checkpoint=torch.load(self.model_path,map_location=device)
        
        #load the weights in the model definition
        model.load_state_dict(checkpoint["state_dict"])
        
        model.eval()# set the model to the evaluation mode
        print("Model is loaded from the disc")
        print("model is:",model)
        return model 
    
    def export_model_to_onnx(self,model):
        
        """
        model is taken as input to this function 
        and the model is exported to the onnx version 
        
        gray is a flag which is False by default. 
        if gray=True then the input model is converted to the grayscale version
        """
        
        #check if the model already exists then no need to export the model to onnx version
        export_path=os.path.join(self.export_path,self.export_name)
        #if os.path.exists(export_path):
        #    print("Model already exists")
        #    print("Not exporting the model again and exiting")
        #    return export_path
        
        if self.gray==True:
            channel=1
            #print("\n Making the gray version of the model \n ")
            #if self.channel!=1:
            #    print("setting the channel value to 1 instead of 3")
            #    print()
            #    self.channel=1
            model=self.convert_model_to_gray(model)
        else:
            channel=3
        
        batch_size=1
        #making a random tensor for the model export
        x=torch.ones(batch_size,channel,self.h_w_shape[0],self.h_w_shape[1],requires_grad=True).to(device) #where channel is local variable 
        #decided by the gray flag

        #make the export path for the model
        export_path=os.path.join(self.export_path,self.export_name)

        #export the model
        torch.onnx.export(model,                       # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    export_path,               # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=14,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input': {0 : 'batch_size'},    # variable length axes
                                    'output': {0 : 'batch_size'}}
                        )
        print("The model is exported to the ONNX version")
        #Loading and running an inference to check the model
        onnx_model=onnx.load(export_path)
        #onnx_model_graph=onnx_model.graph
        onnx_session=onnxruntime.InferenceSession(onnx_model.SerializeToString())
        input_shape=(batch_size,channel,self.h_w_shape[0],self.h_w_shape[1])
        x=torch.randn(input_shape).numpy()
        print("shape of x is:",x.shape)
        #input_names=["input"]
        #output_names=["output"]
        onnx_output=onnx_session.run(['output'],{'input':x})[0]
        return export_path

    def preprocess_image(self):
        
        """
        this function preprocess the image based on the channel and the 
        h_w_shape information to feed the model.
        """
        
        if self.gray==True:
            print("Converting the image to grayscale")
            transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.Resize((self.h_w_shape[0],self.h_w_shape[1])),
                                          transforms.ToTensor()])
        else:
            transform=transforms.Compose([transforms.Resize((self.h_w_shape[0],self.h_w_shape[1])),
                                          transforms.ToTensor()])
            
        image=Image.open(self.image_path)
        image=transform(image)
        image=image.unsqueeze(0)
        image=image.to("cpu")
        
        image=np.array(image)
        
        print("type of image:",type(image),"shape of image:",image.shape)
        
        return image
    
    def postprocess_image(self,model_output):
        
        """
        this function postprocess the image based on the channel and the 
        h_w_shape information to feed the model.
        """
        
        img_format="WEBP"
        img_qual=self.second_quality
        
        #command to convert the pytorch tensor to PIL image
        print("model output type is:",type(model_output))
        print("model_output shape:",model_output.shape)
        
        #remove the first dimension of the model_output numpy array
        model_output=model_output.squeeze(0)
        print("model output shape:",model_output.shape)
        
        if self.gray==False:
            model_output=model_output.transpose(1,2,0)
            image=Image.fromarray(np.uint8(model_output*255))
        else:
            #if grayscale then transpose not required
            #since the gray scale model also output the shape of (3,height,width)
            #get the first dimension only
            model_output=model_output[0]
            model_output=np.reshape(model_output,(self.h_w_shape[0],self.h_w_shape[1]))
            image=Image.fromarray(np.uint8(model_output*255),mode="L")
        
        if self.gray==True:
            image_name=self.model_name+"_"+str(self.quality)+"_GRAY"+"."+img_format.lower()
        else:
            image_name=self.model_name+"_"+str(self.quality)+"."+img_format.lower()
        image.save(image_name,format=img_format,quality=img_qual)
        
        print("The image is saved as :",image_name)
              
    def get_onnx_model_info(self,onnx_model_path):
        """Gives the basic information of the model
        """
        
        model=onnx.load(onnx_model_path)
        from onnx import numpy_helper
        graph=model.graph
        print("\nModel Information:")
        
        # Input and Output Tensor shapes
        print("\nInput Tensor Information:")
        for input_tensor in graph.input:
            shape = []
            tensor_type = input_tensor.type.tensor_type
            for dim in tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else 'dynamic')
            print(f"  - Input Name: {input_tensor.name}")
            print(f"  - Input Shape: {shape}")
            
        input_shape=shape

        print("\nOutput Tensor Information:")
        for output_tensor in graph.output:
            shape = []
            tensor_type = output_tensor.type.tensor_type
            for dim in tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else 'dynamic')
            print(f"  - Output Name: {output_tensor.name}")
            print(f"  - Output Shape: {shape}")
            break #get the first tensor only 

        output_shape=shape
        return input_shape,output_shape
    def convert_model_to_gray(self,model):
        """
        this function takes the model as input and 
        converts it to grayscale version.
        This function takes the model and modifies the first layer of the encoder
        """ 
        #extract the first conv layer 
        first_conv=model.g_a[0]
        
        #create a new conv2d layer with single input channel
        new_conv=nn.Conv2d(1,first_conv.out_channels,kernel_size=first_conv.kernel_size,stride=first_conv.stride,padding=first_conv.padding)
        
        #Initialize new weights for the grayscale input
        with torch.no_grad():
            #Average the original weights across the input channels to match the 1 channel input
            new_conv.weight[:]=first_conv.weight.mean(dim=1,keepdim=True)
            new_conv.bias[:]=first_conv.bias
            
        #replace the old conv layer with the new one
        model.g_a[0]=new_conv
        
        
        params=sum(p.numel() for p in model.parameters())
        print("Parameters in the gray converted model are:",params)
        #return the new model
        model=model.to(device)
        return model
    
    def convert_model_to_gray_new(self, model):
        """
        Convert the model to grayscale version.
        Modifies both the first Conv2d layer of the encoder and the last ConvTranspose2d layer of the decoder
        to ensure both input and output are grayscale.
        """
        # Extract the first conv layer (encoder)
        first_conv = model.g_a[0]
        
        # Create a new Conv2d layer with a single input channel for grayscale input
        new_conv = nn.Conv2d(1, first_conv.out_channels, 
                            kernel_size=first_conv.kernel_size, 
                            stride=first_conv.stride, 
                            padding=first_conv.padding)
        
        # Initialize new weights for the grayscale input
        with torch.no_grad():
            # Average the original weights across the input channels to match the 1 channel input
            new_conv.weight[:] = first_conv.weight.mean(dim=1, keepdim=True)
            new_conv.bias[:] = first_conv.bias
        
        # Replace the old conv layer in the encoder with the new one
        model.g_a[0] = new_conv

        # Extract the last conv transpose layer (decoder)
        last_conv = model.g_s[-1]
        
        # Create a new ConvTranspose2d layer with a single output channel for grayscale output
        new_last_conv = nn.ConvTranspose2d(last_conv.in_channels, 1, 
                                        kernel_size=last_conv.kernel_size, 
                                        stride=last_conv.stride, 
                                        padding=last_conv.padding, 
                                        output_padding=last_conv.output_padding)
        
        # Initialize new weights for the grayscale output
        with torch.no_grad():
            # The original weights are of shape [out_channels, in_channels, kernel_height, kernel_width]
            # We want to average across the output channels (dim=0) to produce 1 output channel
            averaged_weights = last_conv.weight.mean(dim=0, keepdim=True)  # average across output channels
            
            # Debugging: Print the shape and size of averaged_weights
            print(f"Averaged weights shape: {averaged_weights.shape}, size: {averaged_weights.numel()}")
            
            # Make sure to check the size before reshaping
            expected_shape = (1, last_conv.in_channels, last_conv.kernel_size[0], last_conv.kernel_size[1])
            
            if averaged_weights.numel() != expected_shape[0] * expected_shape[1] * expected_shape[2] * expected_shape[3]:
                print(f"Warning: The number of elements {averaged_weights.numel()} does not match the expected size {expected_shape}")
            
            # Reshape the averaged weights to match the new last ConvTranspose2d layer
            new_last_conv.weight[:] = averaged_weights.view(*expected_shape)
            
            # Assign the bias to the new layer
            new_last_conv.bias[:] = last_conv.bias.mean(dim=0, keepdim=True)  # Average biases if needed
        
        # Replace the old conv transpose layer in the decoder with the new one
        model.g_s[-1] = new_last_conv

        # Print model parameter count for verification
        params = sum(p.numel() for p in model.parameters())
        print("Parameters in the gray converted model are:", params)

        # Move model to the appropriate device and return
        model = model.to(device)
        return model
                    
    def run_inference(self):
        input_image=self.preprocess_image()
        
        #load the onnx model
        session_options=onnxruntime.SessionOptions()
        
        #disable the optimization for timebeing
        #session_options.graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        session_options.graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        #check if CUDA is available
        providers = [('CUDAExecutionProvider',{"use_tf32":0})] if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']
        print("Using:",providers)
        
        #model path is the compound of model_path and model_name
        model_path=os.path.join(self.export_path,self.export_name)
        onnx_session = onnxruntime.InferenceSession(model_path,sess_options=session_options, providers=providers)
        
        # Run inference
        input_names = ["input"]
        output_names = ["output"]
        start=time.time()
        onnx_output = onnx_session.run(output_names, {input_names[0]: input_image})[0]
        end=time.time()

        print("Time in seconds for the image inference is:",round(end-start,2),"seconds")
        
        onnx_output=np.clip(onnx_output,0,1)
        print("Onnx output shape is:",onnx_output.shape)
        
        self.postprocess_image(onnx_output)
        
    def main(self):
        
        
        #first loading and exporting the model to onnx version using the appropriate source of the model 
        export_path=os.path.join(self.export_path,self.export_name)
        if os.path.exists(export_path) and self.gray==False:
            print("then re-export the model in the gray version")
            model=self.get_the_model()
            export_path=self.export_model_to_onnx(model)
            
            #now load the model from the export path and give the information about the model
            input_shape,output_shape=self.get_onnx_model_info(export_path)
            print()
            if self.inference:
                self.run_inference()
            return export_path,input_shape,output_shape
            
            
            
        elif os.path.exists(export_path) and self.gray==True:
            print("then re-export the model in the gray version")
            model=self.get_the_model()
            export_path=self.export_model_to_onnx(model)
            
            #now load the model from the export path and give the information about the model
            input_shape,output_shape=self.get_onnx_model_info(export_path)
            print()
            if self.inference:
                self.run_inference()
            return export_path,input_shape,output_shape
        
        elif os.path.exists(export_path)==False:
            print("exporting the model in the gray version according to the gray flag")
            model=self.get_the_model()
            export_path=self.export_model_to_onnx(model)
            
            input_shape,output_shape=self.get_onnx_model_info(export_path)
            print()
            if self.inference:
                self.run_inference()
            return export_path,input_shape,output_shape
            
        elif self.model_path:
            print()
            print("Model path is provided hence loading the model from the disc")
            if self.gray==True:
                print("exporting the model to gray version")
            else:
                print("exporting the model to the RGB version")
            model=self.load_model()
            export_path=self.export_model_to_onnx(model)
            #now load the model from the export path and give the information about the model
            input_shape,output_shape=self.get_onnx_model_info(export_path)
            if self.inference:
                self.run_inference()
            return export_path,input_shape,output_shape
            
        return export_path
    


if __name__=="__main__":

    #make the parser object
    parser=argparse.ArgumentParser(description="Export the Image compression PyTorch model to the ONNX")

    parser.add_argument("--model_name",type=str,required=False,help="choose among"+str(list(models.keys())))
    parser.add_argument("--quality", type=int, default=4, help="choose among 1 to 8 for some models, 1 to 6 for some other models")
    parser.add_argument("--metric", type=str, default="ms-ssim", help=" choose ms-ssim or mse ")
    
    parser.add_argument("--image_path",type=str,required=False,help="Path to the image file")
    
    parser.add_argument("--h_w_shape", type=int, nargs=2, default=[1232, 1640], help="input image [height,width]  like [720,1280]")
    parser.add_argument(
        "--export_path",
        type=str,
        default=os.getcwd(),
        help="Path to export the ONNX model",
    )
    parser.add_argument(
        "--export_name",
        type=str,
        default="model.onnx",
        help="Name of the exported ONNX file with the .onnx extension",
    )

    parser.add_argument("--model_path",type=str,default=None,help="Path to the model checkpoint.pth.tar file")
    parser.add_argument("--gray",action="store_true",help="convert the model to grayscale")
    parser.add_argument("--inference",action="store_true",help="run the inference on the model")
    parser.add_argument("--second_quality",type=int,required=False,help="Give values in between the 0 to 100 to determine the secondary encoder's quality parameter")
    args=parser.parse_args()
    onnx_version=ONNX_version(args)
    export_path,input_shape,output_shape=onnx_version.main()
    

    #onnx_version.preprocess_image()
    #onnx_version.run_inference()
    #onnx_version.onnx_model_info()
    #model=onnx_version.load_model()
    
    
    
