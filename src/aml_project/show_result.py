from  aml_project import model, dataset, preprocess
import aml_project.model


# def mask_batch(batch: torch.Tensor, config:Config, device: torch.device):
#     assert config.resolution == batch.shape[2:], "batch is of an incorrect resolution"
#     ell = [sample_ellipses_mask(config.resolution, config.num_ellipses_train, device=device) for i in range(batch.shape[0])]
#     ell = torch.stack(ell, dim=0).unsqueeze(1)
#     masked = (ell*(batch))
#     masked = torch.cat((masked, ell), 1)
#     return masked

from aml_project.view import view_images
from aml_project.save import save_images
from aml_project.model import mask_batch
from config import Config
import torch
from torch import nn

def main():
    criterion = nn.L1Loss()
    config = Config()
    device = config.detect_device()
    preprocessor = preprocess.Processor(config)
    with torch.no_grad():
        model = aml_project.model.Photosciop(config).to(device)
        model.load_state_dict(torch.load("data/model_weights", map_location=device))
        ds = dataset.ImageOnlyDataset(config, "validation")
        index = torch.randint(0, len(ds)+1, ()).item()
        img = preprocessor(ds[index].unsqueeze(0)).to(device)
        batch = img
        inputs = mask_batch(batch, config, device)
        inp = inputs[:, :3, :, :]
        targets = batch
        out = model(inputs)
        mask = inputs[:, 3:4, :, :]
        
        masked_outputs = out * (1 - mask) + img * mask
        # masked_outputs = out
        loss = (masked_outputs - targets).abs().sum() / (1 - mask).sum()

        
        images_to_plot = [ inputs[0], (masked_outputs).squeeze(0), targets[0]]
        images_to_plot = preprocessor.denorm(images_to_plot)
        # images_to_plot = (torch.cat((inputs[:, :3, :, :], model(inputs), img)))
        view_images(images_to_plot, ["output", "targets"])
        save_images(images_to_plot, ["output", "targets"],"data/out.jpg")

        

if __name__ == "__main__":
    main()