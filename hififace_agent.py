# Copyright 2024 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
import time
from typing import Optional
from configs.train_config import TrainConfig
from models.model import HifiFace
from benchmark.app_image import ImageSwap


import cv2
# import insightface
import numpy as np
# from insightface.app import FaceAnalysis
from livekit import agents, rtc
from livekit.plugins import deepgram
from openai import AsyncOpenAI
import argparse



INTRO_MESSAGE = """
Hi there! Welcome to the demo! Your conversation will be summarized here every minute!
"""

BYE_MESSAGE = """
Thanks for giving this a try! To keep testing, please reload the page and reconnect. Goodbye for now.
"""

_OUTPUT_HEIGHT = 320
_OUTPUT_WIDTH = 240
DET_DIM = 128
FPS_LIMIT = 8
NUM_SWAPPERS = 1
SUMMARIZATION_INT_SEC = 60
END_SESSION_AFTER_MIN = 35
DROP_MODEL_AFTER_NO_PARTICIPANTS_LEFT_SEC = 10


class DeepfakeSTT:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        instance = DeepfakeSTT(ctx)
        await instance.start()

    def __init__(self, ctx: agents.JobContext):
        self.ctx: agents.JobContext = ctx
        # self.detector = FaceAnalysis(name="buffalo_l")
        # self.swappers = [
        #     insightface.model_zoo.get_model(
        #         "inswapper_128.onnx", download=False, download_zip=False
        #     )
        #     for i in range(NUM_SWAPPERS)
        # ]

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
        self.source_face = None

        
        
        
        # self.source_face = None

        self.video_out = rtc.VideoSource(_OUTPUT_WIDTH, _OUTPUT_HEIGHT)
        self.latest_results = []
        self.detecting = False
        self.chat = rtc.ChatManager(ctx.room)
        # STT
        self.speaking_participants = {}
        self.prompt: Optional[str] = None
        self.transcribed_text = ""
        self.last_sent_time = time.time()
        # FPS
        self.start_time = None
        self.frame_count = 0
        # logging.info(f"{NUM_SWAPPERS} swappers spawned.")
        logging.info(f"fps limit: {FPS_LIMIT}")
        # openai Summarization
        self.summarizer = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def start(self):
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            # agent subscribes to patient track:
            if (
                track.kind == rtc.TrackKind.KIND_VIDEO
                and participant.metadata == "patient"
            ):
                self.ctx.create_task(self.process_video_track(track))
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                self.ctx.create_task(self.audio_track_worker(track, participant.sid))

                if isinstance(track, rtc.RemoteAudioTrack):
                    self.speaking_participants[participant.sid] = True

                    print(f"participant{str(self)}")
                    print(
                        f"Participant with ID '{participant.sid}' subscribed to audio track '{track.sid}'"
                    )
                else:
                    # Handle other track types differently (optional)
                    print(f"Received unexpected track type: {type(track)}")

        # take the source face for faceswap
        image_path = "faces/em.jpeg"
        # source_img = cv2.imread(image_path)
        self.source_face = cv2.imread(image_path)
        # self.source_face = np.asarray(self.source_face)
            # source_face = np.asarray(source_face)
            # target_face = np.asarray(target_face)


        # self.detector.prepare(ctx_id=0, det_size=(DET_DIM, DET_DIM))
        # self.source_face = self.detector.get(source_img)[0]

        self.ctx.room.on("track_subscribed", on_track_subscribed)

        video_track = rtc.LocalVideoTrack.create_video_track(
            "agent-video", self.video_out
        )
        await self.ctx.room.local_participant.publish_track(video_track)
        # Send an empty frame to initialize the video track
        result = rtc.VideoFrame(
            _OUTPUT_WIDTH,
            _OUTPUT_HEIGHT,
            rtc.VideoBufferType.ARGB,
            bytearray(_OUTPUT_WIDTH * _OUTPUT_HEIGHT * 4),
        )
        self.video_out.capture_frame(result)

        self.update_state("idle")

        # give time for the subscriber to fully subscribe to the agent's tracks
        await asyncio.sleep(1)
        await self.chat.send_message(INTRO_MESSAGE)
        self.ctx.create_task(self.end_session_after(END_SESSION_AFTER_MIN * 60))
        self.ctx.room.on("participant_disconnected", self.on_participant_disconnected)

    def on_participant_disconnected(self, event):
        print("Participant disconnected")
        participants_count = len(self.ctx.room.participants.keys())
        print(participants_count)
        # if participants_count == 0:
            # del self.detector
            # del self.swappers

    #     self.ctx.create_task(
    #         self.drop_model_after_participants_left(
    #             DROP_MODEL_AFTER_NO_PARTICIPANTS_LEFT_SEC
    #         )
    #     )

    # # drop the model if nobody is connected to the room.
    # async def drop_model_after_participants_left(self, interval: int = 1):
    #     while True:
    #         participants_count = len(self.ctx.room.participants.keys())
    #         if participants_count == 0:
    #             del self.detector
    #             del self.swappers
    #             await self.ctx.disconnect()
    #             return
    #         else:
    #             await asyncio.sleep(interval)

    async def end_session_after(self, duration: int):
        await asyncio.sleep(duration)
        await self.chat.send_message(BYE_MESSAGE)
        # del self.detector
        # del self.swappers
        self.update_state("idle")
        await asyncio.sleep(5)
        await self.ctx.disconnect()

    async def audio_track_worker(self, track: rtc.Track, participant_id: str):
        stt = deepgram.STT()
        stt_stream = stt.stream()
        audio_stream = rtc.AudioStream(track)

        self.ctx.create_task(self.process_text_from_speech(stt_stream, participant_id))
        async for audio_frame_event in audio_stream:
            stt_stream.push_frame(audio_frame_event.frame)
        await stt_stream.flush()

    async def process_text_from_speech(self, stream, participant_id):
        async for event in stream:
            if not event.is_final:
                # received a partial result, STT result be updated as confidence increases
                continue
            if event.alternatives:
                # for alt in event.alternatives:
                #     print(alt.text)
                first_alternative = event.alternatives[0]
                recognized_text = first_alternative.text
                if participant_id in self.ctx.room.participants:
                    speaking_participant_name = self.ctx.room.participants[
                        participant_id
                    ].name
                    # for participant_id, is_speaking in self.speaking_participants.items():
                    #     if is_speaking:
                    #         speaking_participant = participant_id
                    #         speaking_participant_name = self.ctx.room.participants[
                    #             participant_id
                    #         ].name
                    #         # break
                    # print(speaking_participant_name)
                    # print("*"*20)
                    if len(recognized_text):
                        # await self.chat.send_message(
                        #     speaking_participant_name + ": " + recognized_text
                        # )
                        self.transcribed_text += (
                            f"{speaking_participant_name}: {recognized_text}\n"
                        )
                        current_time = time.time()
                        time_elapsed = current_time - self.last_sent_time
                        if time_elapsed >= SUMMARIZATION_INT_SEC and len(
                            self.transcribed_text
                        ):
                            await self.summarize_and_send(self.transcribed_text)
                            self.transcribed_text = ""
                            self.last_sent_time = current_time
                    else:
                        # print("No speaker identified for recognized text.")
                        message = f"(Unknown): {recognized_text}"  # Handle unidentified speaker
            else:
                print("No recognized text found in the event.")
                pass
        await stream.aclose()

    # async def process_text_from_speech(self, stream):
    #     async for event in stream:
    #         if not event.is_final:
    #             # received a partial result, STT result be updated as confidence increases
    #             continue
    #         if event.alternatives:
    #             for alt in event.alternatives:
    #                 print(alt.text)
    #             first_alternative = event.alternatives[0]
    #             recognized_text = first_alternative.text
    #             speaking_participant = None
    #             speaking_participant_name = None
    #             for participant_id, is_speaking in self.speaking_participants.items():
    #                 if is_speaking:
    #                     speaking_participant = participant_id
    #                     speaking_participant_name = self.ctx.room.participants[
    #                         participant_id
    #                     ].name
    #                     # break
    #                     print(speaking_participant_name)
    #             print("*"*20)
    #             if speaking_participant and len(recognized_text):
    #                 await self.chat.send_message(
    #                     speaking_participant_name + ": " + recognized_text
    #                 )
    #                 self.transcribed_text += (
    #                     f"{speaking_participant_name}: {recognized_text}\n"
    #                 )
    #                 current_time = time.time()
    #                 time_elapsed = current_time - self.last_sent_time
    #                 if time_elapsed >= SUMMARIZATION_INT_SEC and len(
    #                     self.transcribed_text
    #                 ):
    #                     await self.summarize_and_send(self.transcribed_text)
    #                     self.transcribed_text = ""
    #                     self.last_sent_time = current_time
    #             else:
    #                 # print("No speaker identified for recognized text.")
    #                 message = (
    #                     f"(Unknown): {recognized_text}"  # Handle unidentified speaker
    #                 )
    #         else:
    #             print("No recognized text found in the event.")
    #             pass
    #     await stream.aclose()

    async def summarize(self, message: str):
        chat_completion = await self.summarizer.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Summarize the conversation or monologue: " + message,
                }
            ],
            model="gpt-3.5-turbo",
        )
        return chat_completion.dict()["choices"][0]["message"]["content"]

    async def summarize_and_send(self, message: str):
        summarized_text = await self.summarize(message)
        await self.chat.send_message(summarized_text)

    # async def process_text_from_speech(self, stream):
    #     async for event in stream:
    #         if not event.is_final:
    #             # received a partial result, STT result be updated as confidence increases
    #             continue
    #         if event.alternatives:
    #             first_alternative = event.alternatives[0]
    #             recognized_text = (
    #                 first_alternative.text
    #             )  # Adjust this attribute access as necessary
    #             speaking_participant = None
    #             # speaking_participant_identity = None
    #             for participant_id, is_speaking in self.speaking_participants.items():
    #                 if is_speaking:
    #                     speaking_participant = participant_id
    #                     # speaking_participant_identity = self.ctx.room.participants[participant_id].identity
    #                     break

    #             if speaking_participant and len(recognized_text):
    #                 await self.chat.send_message(recognized_text)
    #             else:
    #                 # print("No speaker identified for recognized text.")
    #                 message = (
    #                     f"(Unknown): {recognized_text}"  # Handle unidentified speaker
    #                 )
    #         else:
    #             print("No recognized text found in the event.")
    #             pass
    #     await stream.aclose()

    # async def process_video_track(self, track: rtc.VideoTrack):
    #     video_stream = rtc.VideoStream(track)
    #     wrapped_video_stream = LatestFrameVideoStreamWrapper(video_stream, max_fps=FPS_LIMIT)

    #     start_time = time.time()
    #     frame_count = 0

    #     async for frame in wrapped_video_stream:
    #         # videoFrame to opencv BGR image
    #         opencv_image = frame.frame.convert(rtc.VideoBufferType.RGB24).data.tobytes()
    #         opencv_image = np.ndarray(
    #             buffer=opencv_image,
    #             dtype=np.uint8,
    #             shape=(frame.frame.height, frame.frame.width, 3),
    #         )
    #         # detect
    #         latest_results = self.detect(opencv_image)

    #         if not latest_results or len(latest_results) == 0:
    #             continue
    #         # insight face process
    #         opencv_image = self.swapper.get(opencv_image, latest_results[0], self.source_face, paste_back=True)
    #         rgba_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2RGBA)
    #         argb_frame = rtc.VideoFrame(
    #             rgba_image.shape[1],
    #             rgba_image.shape[0],
    #             rtc.VideoBufferType.RGBA,
    #             rgba_image.tobytes(),
    #         )
    #         self.video_out.capture_frame(argb_frame)
    #         frame_count += 1

    #         # Calculate FPS every 5 seconds
    #         start_time, frame_count = self.calculate_fps(start_time, frame_count)

    def calculate_fps(self):
        current_time = time.time()
        if current_time - self.start_time >= 5:
            elapsed_time = current_time - self.start_time
            fps = self.frame_count / elapsed_time
            print(f"FPS: {fps}", end="\r")

            # Reset counters for the next interval
            self.start_time = current_time
            self.frame_count = 0
        return

    async def calculate_fps_continuously(self):
        while True:
            await asyncio.sleep(5)  # Wait for 5 seconds
            self.calculate_fps()

    async def process_video_track(self, track: rtc.VideoTrack):
        video_stream = rtc.VideoStream(track)
        wrapped_video_stream = LatestFrameVideoStreamWrapper(
            video_stream, max_fps=FPS_LIMIT
        )

        self.start_time = time.time()  # Store start time as an attribute
        self.frame_count = 0  # Store frame count as an attribute

        async def process_frame(frame, swapper):
            # source_face = Image.open(args.source_image)
            # target_face = Image.open(args.target_image)

            # source_face = np.asarray(source_face)
            target_face = np.asarray(target_face)

            # Perform inference
            result = image_infer.inference(self.source_face, target_face, args.shape_rate, args.id_rate, args.iterations)

            rgba_image = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
            argb_frame = rtc.VideoFrame(
                rgba_image.shape[1],
                rgba_image.shape[0],
                rtc.VideoBufferType.RGBA,
                rgba_image.tobytes(),
            )
            return argb_frame


            # Save the result
            # result = Image.fromarray(result)


            # videoFrame to opencv BGR image
            # opencv_image = frame.frame.convert(rtc.VideoBufferType.RGB24).data.tobytes()
            # opencv_image = np.ndarray(
            #     buffer=opencv_image,
            #     dtype=np.uint8,
            #     shape=(frame.frame.height, frame.frame.width, 3),
            # )
            # # detect
            # latest_results = self.detect(opencv_image)

            # if not latest_results or len(latest_results) == 0:
            #     return None
            # # insight face process
            # opencv_image = swapper.get(
            #     opencv_image, latest_results[0], self.source_face, paste_back=True
            # )
            # rgba_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2RGBA)
            # argb_frame = rtc.VideoFrame(
            #     rgba_image.shape[1],
            #     rgba_image.shape[0],
            #     rtc.VideoBufferType.RGBA,
            #     rgba_image.tobytes(),
            # )
            # return argb_frame

        async def process_frames():
            # num_swappers = len(self.swappers)

            async def process_single_frame(frame, swapper):
                processed_frame = await process_frame(frame, swapper)
                if processed_frame is not None:
                    self.video_out.capture_frame(processed_frame)
                    self.frame_count += 1

            async def process_frame_with_swapper(frame, swapper_index):
                swapper = self.swappers[swapper_index]
                await process_single_frame(frame, swapper)

            # idx = 0
            # num_swappers = len(self.swappers)
            async for frame in wrapped_video_stream:
                # swapper_index = idx % num_swappers
                # idx += 1
                asyncio.create_task(process_frame_with_swapper(frame, swapper_index))

        # Start the process_frames task
        process_frames_task = asyncio.create_task(process_frames())

        # Start the calculate_fps_continuously task
        calculate_fps_task = asyncio.create_task(self.calculate_fps_continuously())

        # Wait for both tasks to complete
        await asyncio.gather(process_frames_task, calculate_fps_task)

    # def detect(self, frame):
    #     # if (not self.detector) or self.detecting:
    #         # return

    #     self.update_state("detecting")
    #     self.detecting = True
    #     try:
    #         # results = await self.detector.get(img=frame)
    #         # Wrap the synchronous call in run_in_executor
    #         # loop = asyncio.get_event_loop()
    #         # results = await loop.run_in_executor(None, self.detector.get, frame)
    #         results = self.detector.get(frame)
    #         self.latest_results = results
    #     finally:
    #         self.detecting = False
    #         return results

    def update_state(self, state: str):
        metadata = json.dumps({"agent_state": state})
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))


class LatestFrameVideoStreamWrapper:
    def __init__(self, video_stream, max_fps=None):
        self._video_stream = video_stream
        self._latest_frame = None
        self._max_fps = max_fps
        self._frame_interval = 1.0 / max_fps if max_fps is not None else None
        self._last_frame_time = None

    async def __aiter__(self):
        async for frame in self._video_stream:
            if self._frame_interval is None:
                self._latest_frame = frame
                yield frame
            else:
                current_time = self._video_stream._loop.time()
                if (
                    self._last_frame_time is None
                    or (current_time - self._last_frame_time) >= self._frame_interval
                ):
                    self._latest_frame = frame
                    self._last_frame_time = current_time
                    yield frame

    def get_latest_frame(self):
        return self._latest_frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for faceswap")

        await job_request.accept(
            DeepfakeSTT.create,
            identity="DeepfakeSTT_agent",
            auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL,
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(job_request_cb)
    agents.run_app(worker)