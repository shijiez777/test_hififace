import cv2
import time
# from insightface.app import FaceAnalysis
# from insightface import model_zoo


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

    # source_face = Image.open(args.source_image)

    source_face = cv2.imread(args.source_image)
    source_face = cv2.cvtColor(source_face, cv2.COLOR_BGR2RGB)

    source_face = np.asarray(source_face)

    # Open the webcam
    cap = cv2.VideoCapture(0)  # 0 represents the default webcam, you can change it if needed

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get the default video frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(f"Capture resolution:{frame_width}, {frame_height}")

    c = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the original frame
        cv2.imshow('Original', frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_to_save = Image.fromarray(frame)
        # Save the PIL image to a file
        frame_to_save.save(str(c) + "_original.jpg")  # Save as JPEG format

        target_face = frame.copy()
        # target_face = cv2.cvtColor(target_face, cv2.COLOR_BGR2RGB)
        # target_face = np.asarray(target_face)



        result = image_infer.inference(source_face, target_face, args.shape_rate, args.id_rate, args.iterations)

        # Display the processed frame
        cv2.imshow('Processed', result)
        
        result = Image.fromarray(result)
        # Save the PIL image to a file
        result.save(str(c) + ".jpg")  # Save as JPEG format
        c += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()