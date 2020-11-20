import cv2
import os
import re
import time
import math
import json
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class ConvEmotion(nn.Module):
    def __init__(self, c_in, c_hidden_a, c_hidden_b, num_classes):
        super(ConvEmotion, self).__init__()

        self.conv1 = nn.Conv2d(c_in, c_hidden_a, kernel_size=10, stride=2)
        self.conv2 = nn.Conv2d(c_hidden_a, c_hidden_b, kernel_size=7, stride=1)
        self.conv_out = nn.Conv2d(c_hidden_b, num_classes,
                                  kernel_size=7, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(c_hidden_b)
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(self.drop(x)))
        x = self.norm(F.relu(self.conv2(x)))
        x = self.conv_out(self.pool(x))
        return x


class MaartenTorch():
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
                             4: "Neutral", 5: "Sad", 6: "Surprised"}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ConvEmotion(1, 32, 64, 7).to(device).double()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        checkpoint = torch.load("conv_emotion_250.torch")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model = model

    def evaluate_face(self, frame):
        print("MaartenTorch evaluate face")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray ,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y- 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            #print(cropped_img.shape)
            tensor = torch.tensor(cropped_img).permute(0,3,1,2).double()
            #print(tensor.shape)
            prediction = self.model(tensor).reshape(tensor.shape[0], -1).max(-1)[1].int()
            #print("Prediction : ", prediction[0])
            cv2.putText(frame, self.emotion_dict[int(prediction[0])], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)
        return frame