
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

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
        height=self.input_shape[1]
        width=self.input_shape[2]
        #print("H:",height,"W:",width)
        
        image=np.array(cv2.imread(image_path))
        #image=cv2.resize(image,(width,height))
        image=image.astype(np.float32)/255.0
        image=image.transpose(2,0,1)
        image=np.expand_dims(image,axis=0)
        return image.ravel()   
    
    def postprocess_and_save_pil(self,output,output_path):
        """
        This is generating the distorted color of green shade"""
        
        height,width=self.output_shape[1:]
        #print("H",height,"W:",width)

        output=np.clip(output,0,1)
        output=output.reshape(3,height,width)
        output=output.transpose(1,2,0) #covnert to the HWC format
        output=(output*255.0).astype(np.uint8)

        img=Image.fromarray(output)

        if not self.gray:
            img.save(output_path)
        else:
            img=img.convert("L")
            img.save(output_path,quality=20)     
            