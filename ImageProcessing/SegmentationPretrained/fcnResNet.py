class SegmentationPretrained:
    import torchvision
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms
    import cv2
    import matplotlib.patches as patches
    COCO_Instance_Category_Names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                                    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                                    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                                    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                                    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                                    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                                    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                                    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                                    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                                    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']#Able to detect all of these classes
    model = ""
    img_path = ""
    def __init__(self,image_path):#constructor
        self.model = self.torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) #Load Pretrained model
        self.model.eval()
        self.img_path = image_path
        #print("class count: ",len(self.COCO_Instance_Category_Names))

    def get_prediction(self,threshold=0.5):
        img = self.Image.open(self.img_path).convert('RGB')
        transform = self.transforms.Compose([self.transforms.ToTensor()])
        img = transform(img) #make image a Tensor

        pred = self.model([img])
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_class = [self.COCO_Instance_Category_Names[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return pred_boxes, pred_class


    def instance_segmentation(self, threshold=0.5):
        boxes, pred_cls = self.get_prediction(threshold=threshold) #call function
        img = self.cv2.imread(self.img_path)
        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        counter  = 0
        petLst= ['bird', 'cat', 'dog'] #animals able to detect
        for i in pred_cls:
            if i not in petLst: #select just objects in LST(animals)
                counter+=1
            else:
                #print(pred_cls)
                fig, ax = self.plt.subplots() #setup subpolt
                rect = self.patches.Rectangle((boxes[counter][0][0], boxes[counter][0][1]), boxes[counter][1][0] - boxes[counter][0][0], boxes[counter][1][1] - boxes[counter][0][1], linewidth=3, edgecolor='r', facecolor='none')
                ax.add_patch(rect) #draw rect on img
                ax.imshow(img)
                break
        try:
            return img, pred_cls[counter], boxes,counter
        except IndexError:
            print("No pet at this photo! Try again")

#seg = SegmentationPretrained('F:\PYTHON\Projekte\FindTheDog\ClassifyPretrained\img\MenschHund.jpg')
#seg.instance_segmentation()#call function
#plt.show() #plot image