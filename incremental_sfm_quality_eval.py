
import os
import subprocess
import argparse
print("Ensure that the colmap is added to the path and 'colmap' command works in the command line")
print("If not then set the path to COLMAP.bat file in COLMAP_PATH variable")
COLMAP_PATH=r"C:\Swapnil\colmap-x64-windows-cuda"


class COLMAP_EVAL:
    def __init__(self,args,img_path=None,project_path=None):
        """
        project path will contain all the results
        images_path will contain the images dataset
        """
        if project_path is None:
            self.project_path=args.project_path
        else:
            self.project_path=project_path
        if img_path is None:
            self.images_path=args.images_path
        else:
            self.images_path=img_path
            
        self.database_path=os.path.join(self.project_path,"database.db")
        self.output_path=self.project_path
        if os.path.exists(os.path.join(self.output_path,"0")):
            self.output_path_0=os.path.join(self.output_path,"0")
        else:
            self.output_path_0=None #this is the path of the "0" directory created after the results
            
        print("Project path is:",self.project_path)
        print("Images path is:",self.images_path)
        print("Database path is:",self.database_path)
        print("Output path is:",self.output_path)
        print("Output path 0 is:",self.output_path_0)
        
        


    def run_command(self,command):
        """Run a system command and check for errors."""
        if COLMAP_PATH!=None:
            print("Before the command is:",command)
            new_command=os.path.join(COLMAP_PATH,"colmap")
            command=command.replace("colmap",new_command)
            print("new command is:",command)
        process = subprocess.run(command, shell=True)
        if process.returncode != 0:
            raise RuntimeError(f"Command failed: {command}")


    def extract_features(self):
        """Extract features from images."""
        print("Running feature extraction...")
        command = f"colmap feature_extractor --database_path {self.database_path} --image_path {self.images_path}"
        self.run_command(command)
        print("Feature extraction completed.")


    def match_features(self):
        """Match features across images."""
        print("Running feature matching...")
        command = f"colmap exhaustive_matcher --database_path {self.database_path}"
        self.run_command(command)
        print("Feature matching completed.")


    def run_mapper(self):
        """Run COLMAP mapper to generate the map incrementally."""
        print("Running incremental mapping...")
        
        # Construct the basic mapper command
        command = f"colmap mapper --database_path {self.database_path} --image_path {self.images_path} --output_path {self.output_path}"

        # Optionally, include a list of specific images to process
        #if image_list:
        #    command += f" --image_list_path {image_list}"

        self.run_command(command)
        print("Incremental mapping completed.")


    def run_bundle_adjustment(self):
        """Refine the model with bundle adjustment."""
        print("Running bundle adjustment...")
        self.output_path_0=os.path.join(self.output_path,"0")
        command = f"colmap bundle_adjuster --input_path {self.output_path_0} --output_path {self.output_path_0}"
        self.run_command(command)
        print("Bundle adjustment completed.")


    def export_model(self):
        """Export the model in PLY format."""
        print("Exporting model...")
        #output_path=os.path.join(output_path,"0")
        export_path=os.path.join(self.output_path_0,"model.ply")
        command = f"colmap model_converter --input_path {self.output_path_0} --output_path {export_path} --output_type PLY"
        self.run_command(command)
        print("Model exported to:", export_path)


    def analyze_model(self):
        """
        from the given output path , it exports the model summary
        in the export path
        export path must have the text file name and the .txt extension"""
        print("Exporting the summary")
        #output_path=os.path.join(output_path,"0")
        file_name=os.path.join(self.project_path,"summary.txt")
        command=f"colmap model_analyzer --path {self.output_path_0} >> {file_name} 2>&1"# --output_path {export_path_summ}"
        self.run_command(command)
        print("Model summary is exported")
    



    def main(self):
        print("Starting the main function")

        
        if self.output_path_0 is not None and os.path.exists(self.output_path_0) and "cameras.bin" in os.listdir(self.output_path_0):
            print("Model solution exists")
            self.analyze_model()
        else:
            
            # Step 1: Feature Extraction
            
            self.extract_features()

            # Step 2: Feature Matching
            
            self.match_features()

            # Step 3: Incremental Mapping
            
            self.run_mapper()

            # Step 4: Bundle Adjustment
            
            self.run_bundle_adjustment()

            # Step 5: Export Model
            
            self.export_model()

            
            self.analyze_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental SfM with COLMAP")
    parser.add_argument("--project_path",required=True,help="Path to the project where the results will be saved")
    parser.add_argument("--images_path",required=True,help="Path to the images dataset")
##    parser.add_argument("--database_path", required=True, help="Path to the COLMAP database file")
##    parser.add_argument("--image_path", required=True, help="Path to the folder with input images")
##    parser.add_argument("--output_path", required=True, help="Path to the output folder where SfM results will be stored")
##    parser.add_argument("--export_path", required=False, help="Path to export the final 3D model in PLY format")
##    parser.add_argument("--image_list", required=False, help="Optional text file with specific images to process")
##    parser.add_argument("--export_path_summ", required=False, help="Path to export the summary of the model")
    

    # Control flow for each step
##    parser.add_argument("--extract_features", action="store_true", help="Run feature extraction")
##    parser.add_argument("--match_features", action="store_true", help="Run feature matching")
##    parser.add_argument("--run_mapping", action="store_true", help="Run incremental mapping")
##    parser.add_argument("--bundle_adjustment", action="store_true", help="Run bundle adjustment to refine the model")
##    parser.add_argument("--export_model", action="store_true", help="Export the 3D model in PLY format")
##    parser.add_argument("--analyze_model", action="store_true", help="Export the 3D model summary to disc")

    args = parser.parse_args()
    colmap_eval=COLMAP_EVAL(args)
    colmap_eval.main()
    #colmap_eval.analyze_model()

