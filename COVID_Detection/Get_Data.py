import os
import shutil
import numpy as np
import pandas as pd

class MakeDataset():

    def __init__(self, imageFolders):
        self.root = './Dataset/'
        self.targetPath = "./Dataset/CT_Images/"
        self.imageFolders = imageFolders

    def TrainData(self):

        #Create Train Dataset
        for folder in self.imageFolders:
            if folder == "Covid_Xray/":
                if not os.path.exists(self.targetPath + 'Train/Covid'):
                    os.mkdir(self.targetPath + 'Train/Covid')

                covid_metadata = pd.read_csv(self.root + 'CT_metadata.csv')

                for (i,row) in covid_metadata.iterrows():
                    if row['finding'] == 'COVID-19' and row['view']=='PA':
                        img_name = row['filename']
                        img_path = self.root + folder + img_name
                        target_path = self.targetPath + 'Train/Covid/' 
                        
                        if img_name not in os.listdir(target_path):
                            shutil.copy2(img_path,target_path)
                
            else:
                for subfolder in os.listdir(self.root + folder):
                    if not os.path.exists(self.targetPath + 'Train/' + subfolder):
                        os.mkdir(self.targetPath + 'Train/' + subfolder)
                    
                    images_path = self.root + folder + subfolder
                    img_names = os.listdir(images_path)

                    for idx in range(len(img_names)):
                        img_name = img_names[idx]
                        img_path = images_path + '/' + img_name
                        target_path = self.targetPath + 'Train/' + subfolder
                            
                        if img_name not in os.listdir(target_path):
                            shutil.copy2(img_path,target_path)

    def TestData(self):
        
        #create train data 
        trainPath = self.targetPath + 'Train/'
        testPath = self.targetPath + 'Test/'

        for folder in os.listdir(trainPath):
            if not os.path.exists(testPath + folder):
                os.mkdir(testPath + folder)
            
            image_list = os.listdir(os.path.join(trainPath, folder))
            test_size = int(len(image_list)*0.2)
             
            for i in range(test_size):
                img_name = image_list[-(i+1)]
                img_path = os.path.join(trainPath, folder, img_name)
                target_path = os.path.join(testPath, folder)

                shutil.move(img_path,target_path)
