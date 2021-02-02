import Get_Data as Data
from Get_Model import BuildModel


dataset_path = './Dataset/'

def CreateDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)

def Directory():
    CreateDirectory(dataset_path + 'CT_Images/')
    CreateDirectory(dataset_path + 'CT_Images/' + 'Train')
    CreateDirectory(dataset_path + 'CT_Images/' + 'Test')

if __name__ == "__main__":
    #Create some directory
    Directory()

    ImagePaths = ["Chest_Xray/","Covid_Xray/"]
    md = Data.MakeDataset(ImagePaths)
    md.TrainData()
    md.TestData()
    
    VggModel = BuildModel()
    VggModel.GetModel(input_shape=(224,224,3), no_of_class=3, learning_rate=0.0001)
    print(VggModel.ModelLayerInfo())