
"""
    -Evaluation of the reconstructed/compressed image quality
    -Image quality parameters such as SSIM, PSNR, LPIPS are evaluated
    -compression ratio is calculated

    Input:
        -Reconstructed images dataset
        -Original images dataset

    Output:
        -Text file containing the individual values
        -Dataset average values
"""

import os
import torch
from PIL import Image
import numpy as np
import time
import argparse


from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from torchvision import transforms
import lpips
loss_fn_alex=lpips.LPIPS(net="alex")
loss_fn_vgg=lpips.LPIPS(net="vgg")

class Dataset_eval:

    def __init__(self,args):
        self.org_dataset=args.org_dataset_path
        print("Original dataset path:",type(self.org_dataset))
        print(self.org_dataset)
        self.recon_dataset=args.recon_dataset_path

    def images_handling(self,path):
        """
        path to the images dataset
        """

        files=os.listdir(path) #files in the directory
        images=[] #getting only the jpg and png images
        for file in files:
            if file.lower().endswith(".jpeg") or file.lower().endswith(".jpg") \
            or file.lower().endswith(".png") or file.lower().endswith(".webp") \
            or file.lower().endswith(".avif"):
                images.append(file)
        
        print("Images in dataset:",len(images))
        if len(images)!=0:
            img_format=images[0].split('.')[1]
            print("File_type:",img_format)
        else:
            print("No images in the path or different format")
        
        return images, path, img_format

    def convert_gray_3_channel(self,rgb_image):
        """
        Expects rgb_image as pillow object
        Returns the numpy array of 1 channel gray data and other 2 as 0
        
            NOT USEFUL NOW, AS WE HAVE ALREADY CONVERTED THE MODEL TO WORK WITH GRAYSCALE IMAGES
        """
        #convert the rgb to gray
        img=rgb_image.convert("L")
        #convert to numpy array
        img_array=np.array(img)
        #get the shape
        shape=img_array.shape
        #new blank 3 channel array
        new_array=np.zeros((3,shape[0],shape[1]))
        #assign the gray part to first channel
        new_array[0]=img_array
        return new_array

    def get_image_details(self,image):
        """
        input the image as pillow image
        """
        #converting to the numpy array
        image_=np.array(image).transpose()
        shape=list(image_.shape) #check the shape
        res=list(image.size) #resolution
        if len(shape)>=3:
            channel=shape[0] #channel
            res=res
        else:
            channel=1
            res=shape
        print("Image details are: shape:",shape,"Resolution:",res,"Channel:",channel)
        return shape,res,channel
        

    def check_and_match_dataset(self):
        """
        #Assumption: All the images in the dataset will have the same characteristics
        
        -checks the images in the org_dataset shape(color gray), resolution, type (PNG etc)
        -checks the images in the recon_dataset
        -call the resize function, color_conversion option to modify the org image according to
        recon image  

        gray_3_channel: is boolean if True convert_gray_3_channel will be called
        
        
        """
        
        gray=True #if True convert the original image to gray image

        #first for the original images dataset
        images_org,org_path,img_format_org=self.images_handling(self.org_dataset)
        
        image_org=Image.open(os.path.join(org_path,images_org[0])) #using Pillow
        shape_org,res_org,channel_org=self.get_image_details(image_org)
        
        print("Image details of the original dataset are: shape:",shape_org,"Resolution:",res_org,"Channel:",channel_org)
    

        #now for the recon images dataset
        images_recon,recon_path,img_format_recon=self.images_handling(self.recon_dataset)

        image_recon=Image.open(os.path.join(recon_path,images_recon[0])) #using Pillow
        shape_recon,res_recon,channel_recon=self.get_image_details(image_recon)
        
        print("Image details of the reconstructed dataset are: shape:",shape_recon,"Resolution:",res_recon,"Channel:",channel_recon)
        
        if channel_recon==1:
            #its gray image hence convert the original image to gray
            gray=True
        if res_org!=res_recon:
            print("Resolution of the images in the dataset is not same")
            res_new=res_recon  #where the res_new is the new resolution for the original images 
        else:
            res_new=None
            
            
        return res_new,gray

    def evaluate_dataset(self):
        
        #check the dataset resizing and graying requirements
        res_new,gray=self.check_and_match_dataset()
        
        #get the images list 
        org_images,org_path,img_format_org=self.images_handling(self.org_dataset)
        recon_images,recon_path,img_format_recon=self.images_handling(self.recon_dataset)
        
        #check if the images in the org and recon dataset are same in number
        if len(org_images)!=len(recon_images):
            print("Unequal number of images in the original and reconstructed dataset")
        
        #make the list
        cr_list=[]
        ssim_list=[]
        psnr_list=[]
        lpips_list=[]
        
        for org_img,recon_img in zip(org_images,recon_images):
            org_img_path=os.path.join(org_path,org_img)
            recon_img_path=os.path.join(recon_path,recon_img)
            
            #org_img_=Image.open(org_img_path)
            if res_new is not None:
                org_img_=self.convert_image(Image.open(org_img_path),res_new,gray) #PIL image
            else:
                org_img_=Image.open(org_img_path)
            
            recon_img_=Image.open(recon_img_path) #PIL images
            
            cr=self.compression_ratio(org_img_path,recon_img_path)
            ssim_=self.calculate_ssim(org_img_,recon_img_)
            psnr_=self.calculate_psnr(org_img_,recon_img_)
            lpips_=self.calculate_lpips(org_img_,recon_img_)
            cr_list.append(cr)
            ssim_list.append(ssim_)
            psnr_list.append(psnr_)
            lpips_list.append(lpips_)
            
            #print("Compression ratio:",cr,"SSIM:",ssim_,"PSNR:",psnr_,"LPIPS:",lpips_)
            
            print("Done for images:",org_img)
        
        return cr_list,ssim_list,psnr_list,lpips_list,img_format_org,img_format_recon
            
            
            
        
            
        
        
    def convert_image(self,image,res_size,gray):
        """
        Function to modify the image.
        -input: image in the pillow format
        -resize the image in the size of res_size
        -if gray is True then convert to gray image
        
        """
        
        #convert the image to the pillow object
        #image=Image.fromarray(image)
        
        #print("res size is:",type(res_size),res_size)
        
        #convert the  image to gray
        if gray:
            image=image.convert("L")
        
        #resize the image
        image=image.resize(res_size)
        
        return image
        
            

    def compression_ratio(self,img_1_path,img_2_path):
        """
            Assuming the first image is uncompressed version
        """
        print(img_1_path)
        try:
            size_1=os.path.getsize(img_1_path) #getting the size of the image on the disc
        except:
            print("img_1 path not valid")

        try:
            size_2=os.path.getsize(img_2_path) #getting the size of the image on the disc
        except:
            print("img_2 path not valid")

        ratio=size_1/size_2
        #print("Compression ratio is:",ratio)
        return ratio

    def calculate_ssim(self,img_org,img_recon):

        #converting the image to the numpy array
        img_recon=np.asarray(img_recon)
        img_org=np.asarray(img_org)
        
        ssim_value=ssim(img_org,img_recon,win_size=3)
        return ssim_value

    def calculate_lpips(self,img_org,img_recon):
        """
        img_org and img_recon are the path of the 
        original image and the reconstructed image
        """

        #converting the image to the numpy array
        img_recon=np.asarray(img_recon)
        img_org=np.asarray(img_org)

        
        transform=transforms.ToTensor() #convert the image to Tensor 
        #print(img_org.shape,img_recon.shape)

        #converting to the tensors
        img_org=transform(img_org)
        img_recon=transform(img_recon)

        #calculate the loss value
        lpips=loss_fn_alex.forward(img_org,img_recon)
        lpips=lpips.detach().numpy() #to cpu then 
        #print("LPIPS:",lpips) 
        return lpips[0][0][0][0]

    def calculate_psnr(self,img_org,img_recon):

        """
        img_org and img_recon are the path of the 
        original image and the reconstructed image
        """

        #converting the image to the numpy array
        img_recon=np.asarray(img_recon)
        img_org=np.asarray(img_org)

        psnr_value=psnr(img_org,img_recon)
        #print("PSNR:",psnr_value)
        return psnr_value

    def calculate_avg(self,list_):
        """
        calculates the average values of the list
        return as float round to 3 decimal places
        """
        #convert the list to the numpy array
        list_=np.array(list_)
        return round(np.mean(list_),3)
    
    def export_result_to_file(self,cr_list,ssim_list,psnr_list,lpips_list,img_format_org,img_format_recon,path):
        """
        Export the results to the text file
        """
        images_count=len(os.listdir(self.org_dataset))
        #write command to print the current date and time using the time module
        local_time=time.localtime()
        time_name=time.strftime("%Y_%m_%d_%H_%M",local_time)
        if path is None:
            file_name="dataset_eval_result"+str(time_name)+".txt"
        else:
            file_name="dataset_eval_result"+str(time_name)+".txt"
            file_name=os.path.join(path,file_name)
        with open(file_name,"w") as file:
            file.write("Dataset evaluation result\n")
            file.write("Original dataset path:"+self.org_dataset+"\n")
            file.write("Reconstructed dataset path:"+self.recon_dataset+"\n")
            file.write("Original image format:"+img_format_org+"\n")
            file.write("Reconstructed image format:"+img_format_recon+"\n")
            file.write("Images in the dataset:"+str(images_count)+"\n")
            file.write("Time:"+time_name+"\n")
            file.write("cr_list:"+str(cr_list)+"\n")
            file.write("ssim_list:"+str(ssim_list)+"\n")
            file.write("psnr_list:"+str(psnr_list)+"\n")
            file.write("lpips_list:"+str(lpips_list)+"\n")
            
            file.write("Compression ratio:"+str(self.calculate_avg(cr_list))+"\n")
            file.write("SSIM avg:"+str(self.calculate_avg(ssim_list))+"\n")
            file.write("PSNR avg:"+str(self.calculate_avg(psnr_list))+"\n")
            file.write("LPIPS avg:"+str(self.calculate_avg(lpips_list))+"\n")
            
    def main(self,path=None):
        #eval_=Dataset_eval(args.org_dataset,args.recon_dataset)
        cr_list,ssim_list,psnr_list,lpips_list,img_format_org,img_format_recon=self.evaluate_dataset()
        self.export_result_to_file(cr_list,ssim_list,psnr_list,lpips_list,img_format_org,img_format_recon,path)

        
        
        
if __name__=="__main__":
    #recon_dataset_path=r"C:\Swapnil\Narrowband_DRONE\Datasets\script_testing\d1"
    #org_dataset_path=r"C:\Swapnil\Narrowband_DRONE\Datasets\script_testing\d2"
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--org_dataset_path",type=str,help="Path to the original dataset")
    parser.add_argument("--recon_dataset_path",type=str,help="Path to the reconstructed dataset")
    args=parser.parse_args()
    
    
    eval_=Dataset_eval(args)
    eval_.main()
    #eval_.check_and_match_dataset()
    #cr_list,ssim_list,psnr_list,lpips_list,img_format_org,img_format_recon=eval_.evaluate_dataset()
    #eval_.export_result_to_file(cr_list,ssim_list,psnr_list,lpips_list,img_format_org,img_format_recon)
    
