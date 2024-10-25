#importing the modules

import tensorrt as trt
print("tenosorRT version:",trt.__version__)
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
from PIL import Image
import cv2
import os
import argparse
import time
import onnx

class TRT_inference:

    def __init__(self, engine_path,dataset_path,export_data_path,input_shape,output_shape,gray):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.dataset_path=dataset_path
        self.export_data_path=export_data_path
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.gray=gray #boolean
        

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            size = trt.volume(engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings
            bindings.append(int(device_mem))

            # Append to the appropiate input/output list
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        #input_data = input_data.astype(np.float32)  # Ensure the data type matches
        expected_shape = self.inputs[0].host.shape
        #print(f"Input data shape: {input_data.shape}, Expected shape: {expected_shape}")

        # Check if the shape of input_data matches expected shape
        if input_data.size != np.prod(expected_shape):
            raise ValueError(f"Input data size mismatch: {input_data.size} vs {np.prod(expected_shape)}")

        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # Set tensor address
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        #output_tensor = self.outputs[0].host.reshape(input_data.shape)

        #return output_tensor
        return self.outputs[0].host

    def image_handling(self):
        lst=os.listdir(self.dataset_path)
        images=[]
        for image in lst:
            if image.lower().endswith("jpg") or image.lower().endswith("jpeg") or image.lower().endswith("png") or image.lower().endswith("webp"):
                images.append(image)
        print("dataset has:",len(images),"images")

        return images

    def inference_over_dataset(self):
        """
        This method runs the inference over the whole dataset
        and exports the output to the disc
        """
        print("Running the inference_over_dataset")
        images=self.image_handling()

        infer_time_lst=[]
        preprocess_time_lst=[]
        overall_time_lst=[]

        for image in images:

            start_=time.time()
            
            start=time.time()
            image_path=os.path.join(self.dataset_path,image)
            image_=self.preprocess_image(image_path)
            end=time.time()
            preprocess_time=round(end-start,2)
            #print("preprocess is done for:",image,"time:",preprocess_time,"seconds")
            preprocess_time_lst.append(preprocess_time)

            #run inference
            start=time.time()
            output_data=self.infer(image_) #inference step
            end=time.time()
            
            infer_time=round(end-start,2)
            #print("TRT inference time:",infer_time,"s")
            infer_time_lst.append(infer_time)

            #post processing the image
            export_path=os.path.join(self.export_data_path,image.split('.')[0]+".jpg")
            self.postprocess_and_save_pil(output_data,export_path)
            #print("exporting done for :",image)

            end_=time.time()
            overall_time=round(end_-start_,2)
            overall_time_lst.append(overall_time)

        infer_time_lst=np.array(infer_time_lst)
        overall_time_lst=np.array(overall_time_lst)
        print("Average inference time per image is:",infer_time_lst.mean(),"for dataset of size:",infer_time_lst.shape)
        print("Average overall time per image is:",overall_time_lst.mean())

        print()
        print("While exporting the image to the Disc:")
        print("Achievable FPS is:",round(1/overall_time_lst.mean(),3),"for image of resolution:",self.input_shape)

    def preprocess_image(self,image_path):
        #print("starting the preprocessing")
        height=self.input_shape[1]
        width=self.input_shape[2]
        #print("H:",height,"W:",width)
        
        image=Image.open(image_path)
        
        if self.gray:
            #covnert the image to gray scale
            image=image.convert("L")
            #image=np.array(image) #convert to numpy array
            image_shape=list(image.size)
            
            #check if the resize is needed
            if image_shape[0] != width or image_shape[1] != height:
                image = image.resize((width, height),resample=Image.LANCZOS)
            
            # Convert to numpy array and normalize
            image = np.array(image, dtype=np.float32) / 255.0
            # Expand dimensions to (1, 1, H, W) for grayscale
            image = np.expand_dims(image, axis=(0, 1))
            #print("Processed grayscale image shape:", image.shape)
        else:
            #image=np.array(image)
            #print("\ninitially the shape of image is:",image.shape,"\n")
            
            image_shape=list(image.size)
            # Check if resize is needed
            if image_shape[0] != width or image_shape[1] != height:
                image = image.resize((width, height),resample=Image.LANCZOS)
            
            # Convert to numpy array and normalize
            image = np.array(image, dtype=np.float32) / 255.0
            # Transpose to (C, H, W) and expand dimensions to (1, C, H, W)
            image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, axis=0)
            #print("Processed color image shape:", image.shape)
            
        image=image.ravel()
        #print("finally the image shape is:",image.shape)
        #print("preprocessing is done")
        return image   
    
    def postprocess_and_save_pil(self,output,output_path):
        """
        This is generating the distorted color of green shade"""
        #print("starting the postprocesing")
        height,width=self.output_shape[1:]
        #print("H",height,"W:",width)

        #print("the receving output shape is:",output.shape,type(output))
        
        #very important step for the postprocessing which ensures the 
        #colored pixels artefacts are not present
        output=np.clip(output,0,1)
        
        if self.gray==False:
            output=output.reshape(3,height,width)
            
            output=output.transpose(1,2,0) #covnert to the HWC format
            #plt.imshow(output)
            #plt.show()
            output=(output*255.0).astype(np.uint8)

            img=Image.fromarray(output)
            img.save(output_path,quality=60)
        else:
            output=output.reshape(3,height,width) #since currently the model outputs the 3 channel only
            output=output.transpose(1,2,0)
            output=(output*255.0).astype(np.uint8)
            img=Image.fromarray(output).convert("L")
            img.save(output_path,quality=60)
        #print("postprocessing is done")

class Build_engine:
    def __init__(self,onnx_path,engine_path,input_shape,gray):
        self.onnx_path=onnx_path
        self.engine_path=engine_path
        self.height=input_shape[1]
        self.width=input_shape[2]
        self.gray=gray

    def save_engine(self,engine):
        #serialize the engine
        serialized_engine=engine.serialize()

        #save the serialized engine to the file
        with open(self.engine_path,"wb") as f:
            f.write(serialized_engine)
        print("Engine is saved to the disc")

    def load_engine(self):
        logger=trt.Logger(trt.Logger.WARNING)

        #Read the serialized engine from the file
        with open(self.engine_path,"rb") as f:
            serialized_engine=f.read()
        #Deserialize the engine
        runtime=trt.Runtime(logger)
        engine=runtime.deserialize_cuda_engine(serialized_engine)
        return engine

    def build_engine(self):
        print("Building engine")
        print("Setting configurations for the engine")
        logger=trt.Logger(trt.Logger.WARNING) #Warning level for logging messages
        builder=trt.Builder(logger)           #object for engine  building process
        #network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) #allows to specify the batch size of 1 during inference
        network=builder.create_network(1) # for keeping the batch size of 1
        parser=trt.OnnxParser(network,logger) #parser object to parse the ONNX model to TensorRT network

        #create a builder configuration
        config=builder.create_builder_config() #object to hold the configuration settings of the engine

        #Set memory pool limit for the workspace
        print("Memory_pool_limit for optimization:",8,"GB")
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<33) 
        #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30) # 1 GB 31 is 2GB
        #Memory used for temporary data during optimization part
        #Reducing it may speed up the engine building process but may harm the optimization process
        #if it requires more memory
        
        #optimization options
        #----------------------------------------------------------------------------------
        half=False
        int8=False
        if half:
            config.set_flag(trt.BuilderFlag.FP16)
        elif int8:
            config.set_flag(trt.BuilderFlag.INT8)

        #To ensure the model runs in FP32 precision
        config.clear_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.INT8)

        #DLA Deep Learning Accelerator disable
        config.clear_flag(trt.BuilderFlag.GPU_FALLBACK)
        

        #strip weights: create and optimize engine without unncessary weights
        strip_weights=False
        if strip_weights:
            config.set_flag(trt.BuilderFlag.STRIP_PLAN)
        #to remove strip plan from config
        config.flags&=~(1<<int(trt.BuilderFlag.STRIP_PLAN))

        config.clear_flag(trt.BuilderFlag.TF32)

        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        config.clear_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        
        #Parsing the onnx model to the parser object
        with open(self.onnx_path,"rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        print("Parsing the ONNX model done")
        
        #Debugging: Print network inputs and outputs
        print("Number of network inputs:",network.num_inputs)
        print("Number of network Ouputs:",network.num_outputs)

        if network.num_inputs==0:
            print("No inputs found in the network")
            return None

        #set optimization profiles if needed
        input_tensor=network.get_input(0)
        if input_tensor is None:
            print("Error: Input tensor is None")
            return None

        print("Input tensor name:",input_tensor.name)
        profile=builder.create_optimization_profile()
        print("profile created")
        #for each tuple the height and width can be different
        #in this case we have kept them same
        
        if self.gray==False:
            profile.set_shape(input_tensor.name,(1,3,self.height,self.width),
                            (1,3,self.height,self.width),
                            (1,3,self.height,self.width))
        else:
            profile.set_shape(input_tensor.name,(1,1,self.height,self.width),
                            (1,1,self.height,self.width),
                            (1,1,self.height,self.width))
        
        print("profile shape set")
        #set_shape sets the shape of the input tensor. Batch,heigh,width. 3 times is because is gives the
        #minimum, optimal and maximum values for the engine to optimize the inputs to the engine version of the model
        config.add_optimization_profile(profile) #adds the optimization profile to the builder configuration
        print("configs added")

        #Build the engine
        serialized_engine=builder.build_serialized_network(network,config)
        if serialized_engine is None:
            print("Failed to build the serialized network")
            return None

        #Deserialize the engine
        print("Building Engine")
        runtime=trt.Runtime(logger)
        engine=runtime.deserialize_cuda_engine(serialized_engine)
        

        return engine

class TRT_version:
    def __init__(self,args):
        #onnx model path is finalized here
        if args.onnx_model_path.endswith(".onnx"):
            self.onnx_model_path=args.onnx_model_path
        else:
            if args.onnx_model_name.endswith(".onnx"):
                self.onnx_model_path=os.path.join(args.onnx_model_path,args.onnx_model_name)
            else:
                self.onnx_model_path=os.path.join(args.onnx_model_path,args.onnx_model_name+".onnx")
            
        self.engine_name=args.engine_name
            
        if args.engine_name:
            self.engine_path=os.path.join(os.path.dirname(self.onnx_model_path),self.engine_name+".engine")
        else: #same as the onnx model name
            self.engine_path=self.onnx_model_path.split(".")[0]+".engine"
    
        #these shapes depend on the onnx version of the model
        #checking for the input 
        print("type of input_shape argument:",type(args.input_shape))
        print("input shape given is:",args.input_shape)
        input_shape=args.input_shape
        input_shape = tuple(map(int, input_shape.split(',')))
        input_shape=input_shape[1:]
        output_shape=args.output_shape
        output_shape = tuple(map(int, output_shape.split(',')))
        output_shape=output_shape[1:]
        #try:
        #    input_shape=(input_shape)
        #except:
        #    print("Error in converting the input shape argument to list")
        #    return 
        print("input_shape is:",input_shape)
        print("output_shape is:",output_shape)
        #now checking the input shape length
        if len(input_shape)!=3:
            print("Input shape format should be [1,3,720,1280] for 3 channel image of size 720x1280")
            return
        print("\n assuming the input shape and ouput shape is same")
            
        self.input_shape=input_shape #input shape expected by the model [channel,heigh,width]
        self.output_shape=output_shape #output shape produced by the model [channel, height, width]
        
        
        self.channel=args.channel
        
        #dataset path and the address for exporting the resulting images
        self.dataset_path=args.dataset_path
        self.export_data_path=args.export_data_path
        
        #if gray is True then the onnx version of the model is grayscale
        if args.gray:
            self.gray=True
        else:
            self.gray=False
       
    def get_onnx_model_info(self):
        
        model=onnx.load(self.onnx_model_path)
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

        print("\nOutput Tensor Information:")
        for output_tensor in graph.output:
            shape = []
            tensor_type = output_tensor.type.tensor_type
            for dim in tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else 'dynamic')
            print(f"  - Output Name: {output_tensor.name}")
            print(f"  - Output Shape: {shape}")
        
    def trt_main(self):
        
        #getting the information about the onnx model layers of the input and the output layers
        self.get_onnx_model_info()
        
        #loading or building the engine
        trt_build_engine=Build_engine(self.onnx_model_path,self.engine_path,self.input_shape,self.gray)
        #trt_build_engine.build_engine()
        print("Engine path is:",self.engine_path)
        #check if the engine exists on the path
        if os.path.exists(self.engine_path):
            engine=trt_build_engine.load_engine()
            print("Engine is loaded from the disc")
        else:
            print("Engine not found at the path, Building the engine")
            start=time.time()
            engine=trt_build_engine.build_engine()
            end=time.time()
            print("engine is built, Time:",round(end-start,2),"seconds")
            trt_build_engine.save_engine(engine)
            print("Engine is exported to the disc")
        print("====================Engine done=============================")
        #print("Engine time:",round(end-start,2),"seconds")

        #Now running the inference on the whole dataset
    
        #instantiating the TensorRTInference class
        inference_obj=TRT_inference(self.engine_path,self.dataset_path,self.export_data_path,self.input_shape,self.output_shape,self.gray)
        inference_obj.inference_over_dataset()
        
        


        
    

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path",required=True,type=str,help="Path to the onnx model")
    parser.add_argument("--onnxx_model_name",required=False,type=str,help="Name of the onnx model")

    
    parser.add_argument("--engine_name",type=str,help="Name of the engine")
    
    parser.add_argument("--input_shape",type=str,help="Input shape of the model")
    parser.add_argument("--output_shape",type=str,help="Output shape of the model")
    
    parser.add_argument("--channel",type=int,help="Number of channels in the model 1 or 3")
    
    parser.add_argument("--dataset_path",type=str,help="Path to the images dataset for inference from tensorrt model")
    parser.add_argument("--gray",action="store_true",help="If the model is gray scale")
    
    parser.add_argument("--export_data_path",type=str,help="Path to the export the compressed images")
    args=parser.parse_args()
    
    trt_version=TRT_version(args)
    trt_version.trt_main()