import torch
import pandas as pd
from torchvision import transforms
from PIL import Image, ImageOps
import tqdm
import os

# load data from path, csv
class DataSet_Scene(torch.utils.data.Dataset):
    def __init__(self, path, device, **kwargs):
        self.path = path
        self.device = device
        self.mode = kwargs.get('mode', 'train')
        self.size = kwargs.get('size', (150, 150))
        self.ablation1 = kwargs.get('ablation1', False)
        self.to_drop = kwargs.get('to_drop', 5)
        self.load_data()

        

    def load_data(self):
        info = os.path.join(self.path, f'{self.mode}_data.csv')
        self.info = pd.read_csv(info)
        self.imgs = self.info["image_name"].values
        self.labels = self.info["label"].values  

        # load imgs imgs/xxxx.jpg
        imgs_path = os.path.join(self.path, "imgs")
        # showing progress bar
        print(f"Loading {self.mode} images...")
        self.imgs = [self.load_img(os.path.join(imgs_path, img)) for img in self.imgs]
        self.imgs = torch.cat(self.imgs, dim=0)
        self.labels = torch.tensor(self.labels)
        print(f"{self.mode} Images loaded!")


        if self.ablation1:
            # drop all data with label to_drop
            idx = torch.where(self.labels != self.to_drop * torch.ones_like(self.labels))
            self.imgs = self.imgs[idx]
            self.labels = self.labels[idx]

            print("Clearing Data...")
            for idx in tqdm.tqdm(range(self.labels.shape[0])):
                if self.labels[idx] > self.to_drop:
                    self.labels[idx] -= 1
                

        

        
    def load_img(self, img_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img = Image.open(img_path)
        img = ImageOps.pad(img, self.size, color='black')
        img = transform(img)
        img = img.unsqueeze(0)        # 1 c h w
        return img

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        return {
            'image': self.imgs[idx],
            'label': self.labels[idx]
        }
    
    def to(self, device):
        self.device = device
        return self
    
    def __iter__(self):
        for idx in range(len(self)):
            yield {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in self[idx].items()}

    
if __name__ == "__main__":
    dataset = DataSet_Scene('/data/zhangchenyu/ai_image', 'cpu', mode = "train")
    import tqdm
    # iter, showing progress bar
    for data in tqdm.tqdm(dataset):
        print(data['image'].shape, data['label'])

                
                