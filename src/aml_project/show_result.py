from  aml_project import model, dataset, preprocess
from aml_project.model import mask_batch
from aml_project.view import view_images
from config import Config
import torch
from torch import nn

def main():
    criterion = nn.L1Loss()
    config = Config()
    device = config.detect_device()
    preprocessor = preprocess.Processor(config)
    with torch.no_grad():
        model = model.Photosciop(config)
        model.load_state_dict(torch.load("data/model_weights", map_location=torch.device("cpu")))
        ds = dataset.ImageOnlyDataset(config, "validation")
        img = preprocessor(ds[0].unsqueeze(0))
        batch = img
        inputs = mask_batch(batch, config, device)
        inp = inputs[:, :3, :, :]
        out = model(inputs)
        images_to_plot = preprocessor.denorm(torch.cat((inp, out, img, img - out, inp - out) ))
        # images_to_plot = (torch.cat((inputs[:, :3, :, :], model(inputs), img)))
        
        view_images(images_to_plot, ["input", "output", "expected", "out - expect", "inp - out"])
        

if __name__ == "__main__":
    main()