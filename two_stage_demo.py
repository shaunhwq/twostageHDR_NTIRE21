import torch
from modelDefinitions.MKHRD import ResMKHDR, HDRRangeNet
from tqdm import tqdm
import os
import argparse
import numpy as np
import torchvision.transforms as transforms
import cv2


class TwoStageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_net = ResMKHDR()
        self.hdr_rec_net = HDRRangeNet()

    def load_checkpoint(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path)
        self.attention_net.load_state_dict(checkpoint["stateDictEG"])
        self.hdr_rec_net.load_state_dict(checkpoint["stateDictER"])

    def forward(self, ldr_img):
        assert not self.training, "Only use in evaluation mode"

        # Author did detach() and empty_cache(), not sure why
        out_attention = self.attention_net(ldr_img)
        torch.cuda.empty_cache()
        out_hdrrec = self.hdr_rec_net(out_attention.detach())
        torch.cuda.empty_cache()
        return out_hdrrec


def get_model_input(img: np.array, device: str, tv_transforms):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    img = tv_transforms(img)
    img = torch.stack([img])
    img = img.to(device)
    return img

def visualize_model_output(output):
    output = output.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :]
    # Following HDRUNet's method, NTIRE 2021 has already been gamma corrected. Need to transform to linear
    output = output ** 2.24
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt_path", type=str, default="weights/singleShotAtentionFinal_checkpoint.pth")
    args = parser.parse_args()

    assert os.path.exists(args.ckpt_path), "Unable to find pretrained weights"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = TwoStageModel()
    model.load_checkpoint(args.ckpt_path)
    model.eval()
    model.to(args.device)

    image_paths = [os.path.join(args.input_dir, img_path) for img_path in os.listdir(args.input_dir) if img_path[0] != "."]

    img_transform = transforms.Compose([transforms.ToTensor(),])

    with torch.no_grad():
        for img_path in tqdm(image_paths, total=len(image_paths), desc="Running TwoStageHDR..."):
            # Load and transform image
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            in_img = get_model_input(image, args.device, img_transform)
            # Model inference
            model_output = model(in_img)
            # Transform model output and save
            out_hdr = visualize_model_output(model_output)

            # Save output
            new_name = os.path.splitext(os.path.basename(img_path))[0] + ".hdr"
            output_path = os.path.join(args.output_dir, new_name)
            cv2.imwrite(output_path, out_hdr)
