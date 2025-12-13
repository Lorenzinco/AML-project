from  aml_project import model, dataset, preprocess
import aml_project.model
from aml_project.view import view_images
from aml_project.save import save_images
from aml_project.model import mask_batch
from config import Config
import torch
from torch import nn
from sys import argv
from PIL import Image
from torchvision.transforms import Compose, PILToTensor, Resize

def main():
    assert len(argv) > 1, f"Usage : {argv[0]} IMAGE_WITH_ALPHA_CHANNEL"
    config = Config()
    device = config.detect_device()
    preprocessor = preprocess.Processor(config)
    preprocess_1 = Compose((PILToTensor(), Resize(config.resolution)))
    with torch.no_grad():
        model = aml_project.model.Photosciop(config).to(device)
        model.load_state_dict(torch.load("data/model_weights", map_location=device))
        img = preprocess_1(Image.open(argv[1]).convert("RGBA")).unsqueeze(0)
        img = preprocessor(img).to(device)
        mask = img[:, 3:4, :, :][0]
        img_no_alpha = img[:, :3][0]
        img = torch.cat((img_no_alpha * mask + torch.randn_like(img_no_alpha) * (1-mask) * 0.1, mask)).unsqueeze(0)
        out = model(img)
        masked_outputs = out * (1 - mask) + img[:, :3] * mask
        # masked_outputs = out
        images_to_plot = [ img[0][:3], out[0], (masked_outputs).squeeze(0)]
        images_to_plot = preprocessor.denorm(images_to_plot)
        # images_to_plot = (torch.cat((inputs[:, :3, :, :], model(inputs), img)))
        #view_images(images_to_plot, ["output", "targets"])
        save_images(images_to_plot, ["input","output","masked output"],"data/out.jpg")

        

if __name__ == "__main__":
    main()