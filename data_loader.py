from torch.utils.data.dataset import Dataset
import pandas as pd
import torchvision.transforms as T
import cv2

class WireDataset(Dataset):
    def __init__(self, input_csv):
        mapping = {'Parasitic': 0, 'Parassitic' : 0,'Wire_Straight_Defect': 1, 'Wire_Straight_Perfect': 2 , 'Wire_Tilted_Defect' : 3, 'Wire_Tilted_Perfect' : 4, 'Null' : 5}
        csv = pd.read_csv(input_csv, dtype="string")
        self.input = csv.image_path
        self.output = csv.label

        inds = self.output.index[self.output=='Delete'].to_list()
        self.output = self.output.drop(inds)
        self.input = self.input.drop(inds).values
        
        
        self.output = self.output.replace(to_replace = mapping).values
        self.image_size = (178,55)
        self.transform = T.Compose([T.ToPILImage(), T.Resize(self.image_size), T.ToTensor()])
        
        

    def __getitem__(self, index):
        # TODO: Add image loading
        img = cv2.imread(self.input[index], cv2.IMREAD_GRAYSCALE)
        input = self.transform(img)
        return input, self.output[index]
    
    def __len__(self):
        return len(self.input)