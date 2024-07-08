import argparse
from benchmark.app_image import ImageSwap
from configs.train_config import TrainConfig
from models.model import HifiFace
from PIL import Image
import numpy as np

class ConfigPath:
    face_detector_weights = "/home/fatken/data/hififace_pytorch/face_detector_scrfd_10g_bnkps.onnx"
    model_path = ""
    model_idx = 80000
    ffmpeg_device = "cuda"
    device = "cuda"

def main():
    cfg = ConfigPath()
    parser = argparse.ArgumentParser(
        prog="benchmark", description="Image Inference Script", epilog="Inference image swap"
    )
    parser.add_argument("-m", "--model_path", required=True, help="Path to the model checkpoint")
    parser.add_argument("-i", "--model_idx", required=True, type=int, help="Model checkpoint index")
    parser.add_argument("-s", "--source_image", required=True, help="Path to the source image")
    parser.add_argument("-t", "--target_image", required=True, help="Path to the target image")
    parser.add_argument("-o", "--output_image", required=True, help="Path to save the output image")
    parser.add_argument("--shape_rate", default=1.0, type=float, help="Shape similarity rate")
    parser.add_argument("--id_rate", default=1.0, type=float, help="ID similarity rate")
    parser.add_argument("--iterations", default=1, type=int, help="Number of iterations")

    args = parser.parse_args()

    cfg.model_path = args.model_path
    cfg.model_idx = args.model_idx
    opt = TrainConfig()
    checkpoint = (cfg.model_path, cfg.model_idx)
    model = HifiFace(opt.identity_extractor_config, is_training=False, device=cfg.device, load_checkpoint=checkpoint)

    image_infer = ImageSwap(cfg, model)

    # Load source and target images
    source_face = Image.open(args.source_image)
    target_face = Image.open(args.target_image)

    source_face = np.asarray(source_face)
    target_face = np.asarray(target_face)

    # Perform inference
    result = image_infer.inference(source_face, target_face, args.shape_rate, args.id_rate, args.iterations)

    # Save the result
    result = Image.fromarray(result)

    # Save the PIL image to a file
    result.save("output_image.jpg")  # Save as JPEG format

    # print(f"Output image saved to {args.output_image}")

if __name__ == "__main__":
    main()
