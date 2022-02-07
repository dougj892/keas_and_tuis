import streamlit as st
import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import torch.nn.functional as F

def main():
    st.title("Kea vs Tui classifier")

    # load the saved model
    model = load_model()

    uploaded_file = st.file_uploader("Upload a bird photo")

    if uploaded_file:
        # display the uploaded file
        st.image(uploaded_file)
        image = image_loader(uploaded_file)
        predictions = F.softmax(model(image), dim = 1)
        
        # To figure out mapping of classes to bird types I used train_set.dataset.dataset.class_to_idx
        # The rest of the code is necessary to get the first element of the tensor
        prob_kea = predictions.detach()[0][0].item()

        st.write(f'Probabilty kea: {prob_kea*100:.1f}%; Probabilty trui: {(1-prob_kea)*100:.1f}%')


@st.cache
def load_model():
	model = torchvision.models.resnet18(pretrained = False)
	model.fc = torch.nn.Linear(512, 2)
	model.load_state_dict(torch.load('kea_v_tui_model.pth', map_location=torch.device('cpu')))
	model.eval()
	return model


def image_loader(image_path):
    """load image, returns cuda tensor"""
    reg_transform = transforms.Compose([transforms.Resize(255), 
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])])
    image = Image.open(image_path)
    image = reg_transform(image).float()
    image = image.unsqueeze(0) 
    return image

if __name__ == "__main__":
    main()