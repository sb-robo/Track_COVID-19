import os
import numpy as np
import pandas as pd
import Get_Data as Data

dataset_path = './Dataset/'

def CreateDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":
    #Create some directory
    CreateDirectory(dataset_path + 'CT_Images/')
    CreateDirectory(dataset_path + 'CT_Images/' + 'Train')
    CreateDirectory(dataset_path + 'CT_Images/' + 'Test')

    ImagePaths = ["Chest_Xray/","Covid_Xray/"]
    md = Data.MakeDataset(ImagePaths)
    md.TrainData()
    md.TestData()
    print("completed")