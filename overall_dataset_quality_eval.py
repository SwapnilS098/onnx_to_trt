"""
    This script evaluates the complete dataset quality in terms of 
    1. Image quality
        SSIM, PSNR, LPIPS, Compression Ratio
    2. SFM quality
        SFM 3D point cloud model summary
    
    
"""
import os
import dataset_img_quality_eval as img_eval
import incremental_sfm_quality_eval as sfm_eval
import argparse

class Quality_eval:
    def __init__(self,args):
        self.org_dataset_path=args.org_dataset_path
        self.recon_dataset_path=args.recon_dataset_path
        self.project_path=args.project_path
        
        if args.complete:
            self.complete=True
            self.sfm_quality=False
        elif args.sfm_quality:
            self.sfm_quality=True
            self.complete=False
        else:
            self.complete=False
            self.sfm_quality=False
            
            
    def main_new(self,args):
        
        if self.complete==True:
            
            #evaluate the image quality
            img_eval_obj=img_eval.Dataset_eval(args)
            img_eval_obj.main(path=args.project_path)
            
            #evaluate the sfm quality
            
            #First for the reconstructed dataset
            recon_project_path=os.path.join(args.project_path,"recon_project")
            if os.path.exists(recon_project_path):
                pass
            else:
                os.makedirs(recon_project_path)
                
            #rgs.images_path=args.recon_dataset_path
            sfm_eval_obj=sfm_eval.COLMAP_EVAL(args,img_path=args.recon_dataset_path,project_path=recon_project_path)
            #2. for recon images dataset
            sfm_eval_obj.main()
            
            
            #Now for the original dataset
            #args.images_path=args.org_dataset_path
            org_project_path=os.path.join(args.project_path,"org_project")
            if os.path.exists(org_project_path):
                pass
            else:
                os.makedirs(org_project_path)
                
            sfm_eval_obj_=sfm_eval.COLMAP_EVAL(args,img_path=args.org_dataset_path,project_path=org_project_path)
            #first for the orginal dataset
            sfm_eval_obj_.main()
            
            
        elif self.sfm_quality==True:
            
            if args.project_path and args.images_path:
                pass
            else:
                print("Please provide the project path and images path")
                return
            print("Running sfm quality evaluation only")
            sfm_eval_obj=sfm_eval.COLMAP_EVAL(args)
            sfm_eval_obj.main()
            
            
"""
    def main(self):
        
        if args.complete:
            if args.org_dataset_path and args.recon_dataset_path and args.project_path and args.images_path:
                pass
            else:
                print("Please provide all the required arguments")
                return
            
            print("Running img quality and sfm quality evaluation")
            #evaluate the image quality
            img_eval_obj=img_eval.Dataset_eval(args)
            img_eval_obj.main(path=args.project_path)
        
            #evaluate the sfm quality
            
            
            recon_project_path=os.path.join(args.project_path,"recon_project")
            if os.path.exists(recon_project_path):
                pass
            else:
                os.makedirs(recon_project_path)
            #rgs.images_path=args.recon_dataset_path
            sfm_eval_obj=sfm_eval.COLMAP_EVAL(args,img_path=args.recon_dataset_path,project_path=recon_project_path)
            #2. for recon images dataset
            sfm_eval_obj.main()
            
            
            #1 for org dataset
            #args.images_path=args.org_dataset_path
            org_project_path=os.path.join(args.project_path,"org_project")
            if os.path.exists(org_project_path):
                pass
            else:
                os.makedirs(org_project_path)
            sfm_eval_obj_=sfm_eval.COLMAP_EVAL(args,img_path=args.org_dataset_path,project_path=org_project_path)
            
            #first for the orginal dataset
            sfm_eval_obj_.main()
            
            
            
        
        elif args.img_quality:
            if args.org_dataset_path and args.recon_dataset_path:
                pass
            else:
                print("Please provide the original and reconstructed dataset")
                return
            print("Running image quality evaluation only")
            img_eval_obj=img_eval.Dataset_eval(args)
            if args.project_path:
                img_eval_obj.main(args.project_path)
            else:
                img_eval_obj.main()
            
        elif args.sfm_quality:
            if args.project_path and args.images_path:
                pass
            else:
                print("Please provide the project path and images path")
                return
            print("Running sfm quality evaluation only")
            sfm_eval_obj=sfm_eval.COLMAP_EVAL(args)
            sfm_eval_obj.main()
        """

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Overall dataset quality evaluation")
    parser.add_argument("--org_dataset_path",required=False,help="Path to the original dataset")
    parser.add_argument("--recon_dataset_path",required=False,help="Path to the reconstructed dataset")
    parser.add_argument("--project_path",required=False,help="Path to the project where the results will be saved")
    parser.add_argument("--images_path",required=False,help="Path to the images dataset")
    
    parser.add_argument("--complete",action="store_true",help="Run the complete evaluation")
    parser.add_argument("--img_quality",action="store_true",help="Run the image quality evaluation")
    parser.add_argument("--sfm_quality",action="store_true",help="Run the sfm quality evaluation")
    
    args=parser.parse_args()
    
    eval_=Quality_eval(args)
    eval_.main()
    
    
    
