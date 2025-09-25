import asyncio
import websockets
import wave
import json
import numpy as np
import cv2
import os
import shutil
import argparse
from lite_avatar import liteAvatar

WEBSOCKET_URI = "ws://localhost:8765"
RESULT_DIR = 'test_client_results'

async def main():
    parser = argparse.ArgumentParser(description='Test the streaming LiteAvatar service.')
    parser.add_argument('wav_file', type=str, help='Path to the WAV file to stream.')
    parser.add_argument('avatar_id', type=str, help='ID of the avatar to use.')
    args = parser.parse_args()

    # Construct the data_dir from the avatar_id
    data_dir = os.path.join('D:/pei2/lite-avatar/resource/avatar', args.avatar_id)

    # Initialize the liteAvatar class for rendering
    avatar = liteAvatar(data_dir=data_dir, num_threads=1, generate_offline=True)

    # Create result directory
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR)

    async with websockets.connect(WEBSOCKET_URI) as websocket:
        print(f"Connected to {WEBSOCKET_URI}")

        # Open the WAV file
        with wave.open(args.wav_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_channels = wf.getnchannels()
            chunk_size = sample_rate  # 1 second of audio

            print(f"Streaming audio from {args.wav_file}...")

            all_params = []

            async def receive_params():
                while True:
                    try:
                        params_json = await websocket.recv()
                        params = json.loads(params_json)
                        all_params.extend(params)
                        print(f"Received {len(params)} animation parameters.")
                    except websockets.exceptions.ConnectionClosed:
                        break

            receiver_task = asyncio.create_task(receive_params())

            while True:
                audio_frames = wf.readframes(chunk_size)
                if not audio_frames:
                    break

                await websocket.send(audio_frames)
                await asyncio.sleep(0.5)  # Simulate real-time streaming

            # Wait for the receiver to finish
            await websocket.close()
            receiver_task.cancel()

            print("Finished streaming audio.")

            # Now, render the video from the received parameters
            print("Rendering video...")
            tmp_frame_dir = os.path.join(RESULT_DIR, 'tmp_frames')
            os.makedirs(tmp_frame_dir)

            for i, params in enumerate(all_params):
                bg_frame_id = i % avatar.bg_video_frame_count
                mouth_img = avatar.param2img(params, bg_frame_id)
                full_img, _ = avatar.merge_mouth_to_bg(mouth_img, bg_frame_id)
                cv2.imwrite(os.path.join(tmp_frame_dir, f'{i:05d}.jpg'), full_img)

            # Create the video using ffmpeg
            video_path = os.path.join(RESULT_DIR, 'output.mp4')
            cmd = f'ffmpeg -r 30 -i {tmp_frame_dir}/%05d.jpg -i {args.wav_file} -c:v libx264 -pix_fmt yuv420p -y {video_path}'
            os.system(cmd)

            print(f"Video saved to {video_path}")

if __name__ == "__main__":
    asyncio.run(main())
