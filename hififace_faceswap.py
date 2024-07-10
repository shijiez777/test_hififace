# import insightface
# from insightface.app import FaceAnalysis 
import cv2
import redis
from livekit import rtc
import numpy as np
import pickle
import logging
import os
import argparse
from configs.train_config import TrainConfig
from benchmark.app_image import ImageSwap
from models.model import HifiFace
from PIL import Image

# DET_DIM = 128
# BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def process_frame(image_infer, frame, placeholder_frame, source_face, args):
    opencv_image = frame.frame.convert(rtc.VideoBufferType.RGB24).data.tobytes()
    opencv_image = np.ndarray(
        buffer=opencv_image,
        dtype=np.uint8,
        shape=(frame.frame.height, frame.frame.width, 3),
    )
    # detect
    # results = detector.get(opencv_image)
    # if not results or len(results) == 0:
    #     argb_frame = placeholder_frame
    # # insight face process
    # else:

    # source_face = Image.open(args.source_image)
    # target_face = Image.open(args.target_image)

    # source_face = np.asarray(source_face)
    # target_face = np.asarray(target_face)

    # Perform inference
    result = image_infer.inference(source_face, opencv_image, args.shape_rate, args.id_rate, args.iterations)
    if result is None:
        return placeholder_frame

    # Save the result
    # result = Image.fromarray(result)

        # opencv_image = swapper.get(opencv_image, results[0], source_face, paste_back=True)
        # opencv_image = cv2.flip(opencv_image, 1)
    rgba_image = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
    argb_frame = rtc.VideoFrame(
        rgba_image.shape[1],
        rgba_image.shape[0],
        rtc.VideoBufferType.RGBA,
        rgba_image.tobytes(),
    )
    return argb_frame



def process_stream(image_infer, source_face, room_id, redis_cli, placeholder_frame, args):
    channel = room_id
    pubsub = redis_cli.pubsub()
    pubsub.subscribe(channel)

    for message in pubsub.listen():
        if message['type'] == 'message':
            frame = pickle.loads(message["data"])
            argb_frame = process_frame(image_infer, frame, placeholder_frame, source_face, args)
            pickled_data = pickle.dumps(argb_frame)
            redis_cli.publish("processed_channel", pickled_data)
            logging.info("processed_channel")

def main():

    placeholder_img = cv2.imread(os.path.join("/home/fatken/projects/faceswapSTT-agent/faces/placeholder.png"))
    placeholder_img = cv2.cvtColor(placeholder_img, cv2.COLOR_RGB2RGBA)
    placeholder_frame = rtc.VideoFrame(
        placeholder_img.shape[1],
        placeholder_img.shape[0],
        rtc.VideoBufferType.RGBA,
        placeholder_img.tobytes(),
    )



    # detector = FaceAnalysis(name="buffalo_l")
    # detector.prepare(ctx_id=0, det_size=(DET_DIM, DET_DIM))
    # swapper = insightface.model_zoo.get_model(os.path.join(BASE_DIR, "inswapper_128.onnx"), download=False, download_zip=False)
    # image_path = os.path.join("/home/shijiez/data/hififace/faces/em.webp")
    # source_img = cv2.imread(image_path)
    # source_face = detector.get(source_img)[0]



    class ConfigPath:
        face_detector_weights = "/home/shijiez/data/hififace/face_detector_scrfd_10g_bnkps.onnx"
        model_path = ""
        model_idx = 80000
        ffmpeg_device = "cuda"
        device = "cuda"


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

    image_path = os.path.join("/home/shijiez/projects/test_hififace/faces/em.jpeg")
    # source_face = Image.open(image_path)
    source_face = cv2.imread(image_path)






    r = redis.StrictRedis(host='localhost', port=6379, decode_responses=False)
    process_stream(image_infer, source_face, "raw_channel", r, placeholder_frame, args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("/tmp/basic_room.log"), logging.StreamHandler()],
    )
    main()


# Run this with command:
# python hififace_faceswap.py --model_path /home/shijiez/data/hififace/standard_model --model_idx 320000 --source_image em.jpeg --target_image me.jpg --output_image image.jpg --shape_rate 1.0 --id_rate 1.0 --iterations 2




