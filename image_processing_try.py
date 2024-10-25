import numpy as np
import os
import time
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def preprocess_image(image_path,input_shape,resize_algos):
        height=input_shape[1]
        width=input_shape[2]
        print("H:",height,"W:",width)
        image=Image.open(image_path)
        image.resize((width,height),resample=Image.NEAREST)
        
        resized_images=[]
        
        for algo in resize_algos:
            print(f"Resizing the image using: {algo}")
            
            if algo == "Resampling.NEAREST":
                image_resized = image.resize((width, height), resample=Image.NEAREST)
                image_resized.save("nearest.png")
                
            elif algo == "Resampling.BILINEAR":
                image_resized = image.resize((width, height), resample=Image.BILINEAR)
                image_resized.save("bilinear.png")
            elif algo == "Resampling.BICUBIC":
                image_resized = image.resize((width, height), resample=Image.BICUBIC)
                image_resized.save("bicubic.png")
            elif algo == "Resampling.LANCZOS":
                image_resized = image.resize((width, height), resample=Image.LANCZOS)
                image_resized.save("lanczos.png")
            elif algo == "Resampling.BOX":
                image_resized = image.resize((width, height), resample=Image.BOX)
                image_resized.save("box.png")
            elif algo == "Resampling.HAMMING":
                image_resized = image.resize((width, height), resample=Image.HAMMING)
                image_resized.save("hamming.png")   
                
            resized_images.append(image_resized)
            
        num_images=len(resized_images)
        rows=1 if num_images<=3 else 2
        cols=min(3,num_images)
        fig,axes=plt.subplots(rows,cols,figsize=(12,6))
        
        for i,(image,algo) in enumerate(zip(resized_images,resize_algos)):
            row, col=divmod(i,cols)
            axes[row,col].imshow(image)
            axes[row,col].set_title(f"Resized Image using algo: {algo}")
            axes[row,col].axis("off")
        plt.tight_layout()
        plt.show()
        
        #for algo in resize_algos:
        #    print("Resizing the image using:",algo)
        #    image=image.resize((width,height),resample=Image.NEAREST)
        #    Image.resize()
        #    plt.imshow(image)
        #    plt.title("Resized Image using algo:"+algo)
        #    plt.show()
        
        
        #    print("image is loaded as np array of shape:",image.shape,image.dtype)
            #print("image is as follows:",image[0])
        #    print()
        #    plt.imshow(image)
        #    plt.title("Original Image")
        #    plt.show()
            #image=cv2.resize(image,(width,height))
        #    print("Now normalizing the image and changing the datatype")
        #    image=np.array(image)
        #    image=image.astype(np.float32)/255.0
        #    plt.imshow(image)
        #    plt.title("Normalized Image")
        #    plt.show()
        #    print("image is loaded as np array of shape:",image.shape,image.dtype)
            #print("image is as follows:",image[0])
        #    print()
        #    image=image.transpose(2,0,1)
        #    print("now the image shape is:",image.shape)
        #    image=np.expand_dims(image,axis=0)
        #    print("now the image shape is as follows:",image.shape)
        #    image_r=image.ravel()
        #    print("Now the image shape is as follows:",image.shape)
        #    print("lets visualize the image:")
        #return image_r
    
    
def preprocess_image_2(image_path,input_shape,gray):
        print("starting the preprocessing")
        height=input_shape[1]
        width=input_shape[2]
        #print("H:",height,"W:",width)
        
        image=Image.open(image_path)
        
        if gray:
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
            print("Processed grayscale image shape:", image.shape)
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
            print("Processed color image shape:", image.shape)
            
        image=image.ravel()
        print("finally the image shape is:",image.shape)
        print("preprocessing is done")
        return image
    
    
resize_algos=["Resampling.NEAREST", "Resampling.BILINEAR", "Resampling.BICUBIC", "Resampling.LANCZOS",
              "Resampling.BOX", "Resampling.HAMMING"]


def postprocess_and_save_pil(output,output_path,output_shape,gray):
        """
        This is generating the distorted color of green shade"""
        print("starting the postprocesing")
        height,width=output_shape[1:]
        print("H",height,"W:",width)

        print("the receving output shape is:",output.shape,type(output))
        
        #very important step for the postprocessing which ensures the 
        #colored pixels artefacts are not present
        output=np.clip(output,0,1)
        
        if gray==False:
            output=output.reshape(3,height,width)
            
            output=output.transpose(1,2,0) #covnert to the HWC format
            plt.imshow(output)
            plt.show()
            output=(output*255.0).astype(np.uint8)

            img=Image.fromarray(output)
            img.save(output_path,quality=90)
        else:
            output=output.reshape(height,width)
            output=output.transpose()
            output=(output*255.0).astype(np.uint8)
            img=Image.fromarray(output,mode="L")
            img.save(output_path,quality=90)
        print("postprocessing is done")

if __name__=="__main__":
    #input_shape=(3,2464,3280)
    input_shape=(3,720,1280)
    image_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\fine_tune_to_trt\image.png"
    preprocess_image(image_path,input_shape,resize_algos)    
    gray=False
    output=preprocess_image_2(image_path,input_shape,gray)
    output_path=os.path.join(os.path.dirname(image_path),"output_NOW.png")
    output_shape=input_shape
    postprocess_and_save_pil(output,output_path,output_shape,gray)