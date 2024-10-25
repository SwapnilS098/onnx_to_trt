
""" 
    COMPRESSAI environment is required for this 
    Single script to take the inputs as folows:
    -PyTorch model path
    -Or DL model name and parameters
    -image_dataset_path
    -project_path: onnx model, engine model ,exported images, quality results are saved there
    -input_shape: shape of the image for the model to be input crafted height width
    -batch_size: btach size for the model 
    -gray: if the model needs to be made gray then use this as FLAG
    -model_name: name of the model to be saved for the description of the model 
    """
    
    
#importing the onnx model version
import model_export_pytorch_onnx as onnx_model_builder

#importing the trt model builderr
import trt_main as trt_model_builder

import os 
import time
import argparse

class DL_TRT_inference:
    
    def __init__(self,args):
        #first init the onnx model conversion arguments 
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
        
        self.project_path=args.project_path
        if os.path.exists(self.project_path)==False:
            os.makedirs(self.project_path)
        
        self.trt_model_name=args.trt_model_name
        
        self.model_path=args.model_path
        
        if args.gray:
            self.gray=True
        else:
            self.gray=False
            
        #initialize with the None and later it will be set using the onnx conversion model's output
        self.onnx_model_path=None 
        
        self.onnx_model_input_shape=None
        self.onnx_model_output_shape=None
        
        
        self.dataset_path=args.image_dataset_path
               
    def make_args_for_onnx(self):
        
        args_onnx = argparse.Namespace()
        args_onnx.model_name = self.model_name
        args_onnx.quality = self.quality
        args_onnx.metric = self.metric
        args_onnx.h_w_shape = self.h_w_shape
        
        args_onnx.export_path=self.project_path
        #args_onnx.export_name=self.trt_model_name
        if self.trt_model_name.endswith(".onnx") == False:
            args_onnx.export_name=self.trt_model_name+".onnx"
        else:
            args_onnx.export_name=self.trt_model_name
            
        args_onnx.model_path=self.model_path
        if args.second_quality:
            args_onnx.second_quality=self.second_quality
        else:
            args_onnx.second_quality=None
        args_onnx.inference=None
        args_onnx.image_path=None
        args_onnx.gray=args.gray
        
        return args_onnx
    
    def make_args_for_trt(self):
        
        args_trt=argparse.Namespace()
        
        args_trt.onnx_model_path=self.onnx_model_path
        args_trt.onnx_model_name=self.trt_model_name
        args_trt.model_name=self.model_name #same as the trt_model_name
        
        args_trt.engine_name=self.trt_model_name #the engine name is kept same as the onnx model name
        
        #assuming the onnx_model_input_shape is like ["dynamic",3,720,1280]
        self.onnx_model_input_shape[0]=1 #batch size is set to 1
        self.onnx_model_output_shape[0]=1 #batch size is set to 1
        
        
        onnx_model_input_shape=','.join([str(item) for item in self.onnx_model_input_shape])
        onnx_model_output_shape=','.join([str(item) for item in self.onnx_model_output_shape])
        
        
        print("final shape is:",onnx_model_input_shape)
        
        
        
        
        args_trt.input_shape=onnx_model_input_shape #converting to string from list
        args_trt.output_shape=onnx_model_output_shape
        
        if self.gray==True:
            args_trt.channel=1
        else:
            args_trt.channel=3
            
        args_trt.dataset_path=self.dataset_path
        
        args_trt.gray=args.gray
        
        export_data_path=os.path.join(self.project_path,"exported_data")
        if os.path.exists(export_data_path)==False:
            os.makedirs(export_data_path)
            
        args_trt.export_data_path=export_data_path
        
        return args_trt
    
    
    def main(self,args):
        
        #get the arguments for the onnx model builder
        args_onnx=self.make_args_for_onnx()
        
        #make the object of the ONNX_version class
        onnx_version=onnx_model_builder.ONNX_version(args_onnx)
        export_path,input_shape,output_shape=onnx_version.main()
        
        #it is just the directory name and not the complete one
        export_path=os.path.dirname(export_path)
        print("input_shape:",input_shape)
        print("output_shape:",output_shape)
        
        #onnx_model_path is uodated with the onnx model path exported above
        self.onnx_model_path=export_path
        
        #update the onnx model input and output shape
        self.onnx_model_input_shape=input_shape
        self.onnx_model_output_shape=output_shape
        
        print("\n====================\n")
        print("====ONNX model exported at:",export_path)
        print("\n====================\n")
        
        
        args_trt=self.make_args_for_trt()
        
        #make the object of the trt model builder
        trt_version=trt_model_builder.TRT_version(args_trt)
        trt_version.trt_main()
        
        
        
    
    
if __name__=="__main__":
    
    parser=argparse.ArgumentParser(description="Script to perform inference on the ONNX model using the TensorRT")
    parser.add_argument("--model_path",type=str,required=True,help="Path to the ONNX model file")
    parser.add_argument("--model_name",type=str,required=False,help="Name of the model")
    parser.add_argument("--quality",type=int,required=False,help="Quality of the model")
    parser.add_argument("--metric",type=str,required=False,help="Metric to be used for the model")
    
    parser.add_argument("--image_dataset_path",type=str,required=True,help="Path to the image dataset")
    parser.add_argument("--project_path",type=str,required=True,help="Path to the project folder")
    parser.add_argument("--second_quality",type=int,required=False,help="traditional encoder quality")
    
    parser.add_argument("--h_w_shape",type=int,nargs=2,required=True,help="Input shape of the model")
    parser.add_argument("--batch_size",type=int,required=False,help="Batch size for the model")
    parser.add_argument("--gray",action="store_true",help="If the model is to be made grayscale then use this as flag")
    parser.add_argument("--trt_model_name",type=str,required=True,help="Name of the exported tensorRT_onnx_model")    
     
    
    
    args= parser.parse_args()
    obj=DL_TRT_inference(args)
    obj.main(args)
    
    
