import numpy as np

import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torch.optim as optim


from torch.utils.data import Subset,ConcatDataset,random_split




import albumentations as A


class ObjectDetector:

    
    def __init__(self):
        
        # Initialize model with pretrained weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)

        # Modify the first conv layer to accept 2 channels instead of 3
        # original_conv = self.model.backbone.body.conv1
        self.model.backbone.body.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Create new transform with 2-channel normalization
        min_size = 640
        max_size = 800  # very close to 798
        image_mean = [0.485, 0.456]
        # Only first 2 channels of original [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224]
        # Only first 2 channels of original [0.229, 0.224, 0.225]
        self.model.transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std
        )

        # Replace the classifier with a new one
        # for our number of classes (4 + background)
        num_classes = 5  # 4 classes + background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.batch_size = 4

        # Class mapping
        self.int_to_cat = {
            0: "beam_from_ionisation",
            1: "laser_driven_wakefield",
            2: "beam_driven_wakefield", 
            3: "beam_from_background",
        }
    
    def prepare_data(self, X, y=None, batch_size=4):
        # create dataset class
        transformer=transforms.Compose(
    [
        transforms.ToTensor()
    
    ]
)   
        augmentation_transform=A.Compose(
    [
        # Géométriques
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # Photométriques
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
     
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),

     


    ],
    bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'] )
)
        class dataset_LWAS_2(torch.utils.data.Dataset) : 

            def __init__(self,transform,X,Y=None,preload=True,path=None,train=True) :
                if preload : 
                    self.x=X
                    self.y=Y
                    self.transform=transform
                    self.preload=True
                    self.train=train
                
                    

            def __len__(self) : 
                if self.preload : 
                    return len(self.x)
                else: 
                    return len(self.list_x_names)
            
            def process_x_y(self,x,y=None) : 
                        
                        
                        img=x['data']
                        img=self.transform(img).to(device).to(torch.float32) ## Ici, je ne sais pasp ourquoi mais mes images s'ouvrent en float16 : obligé de forcer la conversion en float32
                        #print("shape : ",img.shape)
                        img=img.permute(1,2,0)
                        img_width=img.shape[2]
                        img_height=img.shape[1]
                        if img.dim()==2 : 
                            img.unsqueeze_(0)
                    
                        if (self.preload and self.y is not None) or (not self.preload and self.train):
                            
                                boxes = []
                                labels = []
                                for box in y:

                                    x_center, y_center, width, height = box['bbox']
                                    
                            
                                    x1 = (x_center - width/2) * img_width
                                    y1 = (y_center - height/2) * img_height
                                    x2 = (x_center + width/2) * img_width
                                    y2 = (y_center + height/2) * img_height
                                    
                                
                                
                                    if x1 >= 0 and y1 >= 0 and x2 <= img_width and y2 <= img_height:
                                        if x2 > x1 and y2 > y1:
                                            boxes.append([x1, y1, x2, y2])
                                            labels.append(box['class'] + 1)  

                                if boxes:
                                    target = {
                                        'boxes': torch.FloatTensor(boxes).to(device),
                                        'labels': torch.tensor(labels, dtype=torch.int64).to(device)
                                    }
                                else:
                                    
                                    target = {
                                        'boxes': torch.FloatTensor(size=(0, 4)),
                                        'labels': torch.tensor([], dtype=torch.int64)
                                    }
                            
                                return img, target
                            
                        return img
            
            def __getitem__(self,idx): 
            
                if self.preload: 
                            
                        img=self.x[idx]
                        if self.y is not None : 
                            y=self.y[idx]
                            img,target=self.process_x_y(x,y)
                            return img,target
                        else : 
                            y=None
                            img=self.process_x_y(x,y)
                            return img
                
                
        class dataset_LWAS_2_augmented(torch.utils.data.Dataset) : 

            def __init__(self, transform, augmentation, X, Y=None,preload=True,path=None,train=True) :
                if preload : 
                    self.x=X
                    self.y=Y
                    self.transform=transform
                    self.preload=True
                    self.train=train
                    self.augmentation=augmentation
              
                    

            def __len__(self) : 
                if self.preload : 
                    return len(self.x)
                else: 
                    return len(self.list_x_names)
            
            def process_x_y(self,x,y=None) : 
                        
                        
                        img=x['data']
                        img=self.transform(img).to(device).to(torch.float32) ## Ici, je ne sais pasp ourquoi mais mes images s'ouvrent en float16 : obligé de forcer la conversion en float32
                        #print("shape : ",img.shape)
                        img=img.permute(1,2,0)
                        img_width=img.shape[2]
                        img_height=img.shape[1]
                        
                    
                        if (self.preload and self.y is not None) or (not self.preload and self.train):
                            
                                boxes = []
                                labels = []
                                for box in y:
                                    x_center, y_center, width, height = box['bbox']
                                    
                            
                                    x1 = (x_center - width/2) * img_width
                                    y1 = (y_center - height/2) * img_height
                                    x2 = (x_center + width/2) * img_width
                                    y2 = (y_center + height/2) * img_height
                                    
                                
                                
                                    if x1 >= 0 and y1 >= 0 and x2 <= img_width and y2 <= img_height:
                                        if x2 > x1 and y2 > y1:
                                            boxes.append([x1, y1, x2, y2])
                                            labels.append(box['class'] + 1)  

                                if boxes:
                                    target = {
                                        'boxes': torch.FloatTensor(boxes).to(device),
                                        'labels': torch.tensor(labels, dtype=torch.int64).to(device)
                                    }
                                else:
                                    
                                    target = {
                                        'boxes': torch.FloatTensor(size=(0, 4)),
                                        'labels': torch.tensor([], dtype=torch.int64)
                                    }

                        
                                img=img.numpy()
                    
                                img=np.transpose(img,(1,2,0))
                                transformed=self.augmentation(image=img,bboxes=target['boxes'].tolist(),class_labels=target["labels"].tolist())
                                target_augm=dict()
                                

                                boxes_tensor = torch.FloatTensor(transformed["bboxes"]).to(device)
                                labels_tensor = torch.tensor(transformed["class_labels"], dtype=torch.int64).to(device) 

                                print(type(boxes_tensor),type(labels_tensor))
                                
                                img,target_augm["boxes"],target_augm['labels']=transformed['image'],boxes_tensor,labels_tensor
                                img=self.transform(img).to(device).to(torch.float32)
                        
                                print(111)
                                if img.dim()==2 : 
                                    img.unsqueeze_(0)
                                


                                return img, target_augm
                            
                        return img
            
            def __getitem__(self,idx): 
            
                if self.preload: 
                            
                        img=self.x[idx]
                        if self.y is not None : 
                            y=self.y[idx]
                            img,target=self.process_x_y(x,y)
                            return img,target
                        else : 
                            y=None
                            img=self.process_x_y(x,y)
                            return img
               

        # Create dataset
        dataset = dataset_LWAS_2(transform=transformer,X=X,Y=y,preload=True,path=None,train=True)
        dataset_augmented= dataset_LWAS_2_augmented(transform=transformer, augmentation=augmentation_transform, X=X, Y=y,preload=True,path=None,train=True)
        # Create data loader
        
        train_set_augmented=Subset(dataset_augmented,range(0,800,2))
        
        combined_dataset=ConcatDataset([train_set_augmented,dataset])
        data_loader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=True if y is not None else False,
            collate_fn=lambda x: tuple(zip(*x))
            )
        
        


        
        return data_loader

    def fit(self, X, y):
        # Prepare training data
        data_loader = self.prepare_data(X, y, batch_size=4)

        # Set model to training mode
        self.model.train()

        # Create optimizer
        optimizer = optim.SGD(self.model.parameters(),
                              lr=0.01,
                              momentum=0.9,
                              weight_decay=0.0005)
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95  # Reduce lr by 5% each epoch
        )

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            epoch_loss = 0
            for images, targets in data_loader:
                optimizer.zero_grad()

                # Move input data to the device
                images = [image.to(self.device) for image in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()}
                    for t in targets
                ]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()

                # Add gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += losses.item()
            # Step the scheduler
            scheduler.step()

            # print(f'Epoch {epoch}: Loss = {epoch_loss/len(images)}')

        return self

    def predict(self, X):
        # Set model to evaluation mode
        self.model.eval()

        # Prepare data
        data_loader_test = self.prepare_data(X)

        predictions = []
        with torch.no_grad():
            for batch in data_loader_test:
                # batch shape is [B, C, H, W] where B=4 (batch_size)
                # Process each image in the batch separately
                for single_img in batch:
                    # Add batch dimension back
                    img = single_img.unsqueeze(0)  # [1, C, H, W]
                    pred = self.model(img)[0]
                    img_preds = []
                    # print(f"Boxes: {pred['boxes']}")
                    # print(f"Labels: {pred['labels']}")
                    # print(f"Scores: {pred['scores']}")
                    # Get image dimensions
                    img_height = img[0].shape[1]  # Height is dim 1
                    img_width = img[0].shape[2]   # Width is dim 2
                    for box, label, score in zip(
                        pred['boxes'], pred['labels'], pred['scores']
                    ):
                        if score > 0.25:  # Confidence threshold
                            # Convert box from pixels
                            # to normalized coordinates [0,1]
                            x1, y1, x2, y2 = box.cpu().numpy()
                            # Normalize coordinates
                            x1 = x1 / img_width
                            x2 = x2 / img_width
                            y1 = y1 / img_height
                            y2 = y2 / img_height
                            # Convert from [x1,y1,x2,y2]
                            # to [x_center,y_center,width,height]
                            width = x2 - x1
                            height = y2 - y1
                            x_center = x1 + width/2
                            y_center = y1 + height/2
                            pred_dict = {
                                'bbox': [x_center, y_center, width, height],
                                'class': int(label.cpu().numpy()) - 1,
                                'proba': float(score.cpu().numpy())
                            }
                            img_preds.append(pred_dict)

                    predictions.append(img_preds)

        return np.array(predictions, dtype=object)