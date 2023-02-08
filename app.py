import streamlit as st
from PIL import Image
import torch
import torch.optim as optim
from model import CustomModel
from clf import predict

# class Transforms:
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, img, *args, **kwargs):
#         return self.transforms(image=np.array(img))['image']

def modelsetup(load_path):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 8
    model = CustomModel(num_classes)
    # model.to(device)
    # next(model.parameters()).device

    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.1)

    def load_checkpoint(checkpoint):
        print(f'=> Loading Checkpoint')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    load_checkpoint(torch.load(load_path))
    return model

load_path = "./model/v2.1-custom_model.pth.tar"
model = modelsetup(load_path=load_path)

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Skin Lesion Classification using Deep Lerning Methodologies")
st.write("")

file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    labels = predict(file_up, model)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])