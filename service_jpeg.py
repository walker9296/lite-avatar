import asyncio
import websockets
import json
import numpy as np
import cv2
import base64
from lite_avatar import liteAvatar

class LiteAvatarService:
    def __init__(self):
        self.avatar = liteAvatar(data_dir='D:/pei2/lite-avatar/data/sample_data', num_threads=1, generate_offline=True)
        self.frame_index = 0

    async def handle_connection(self, websocket, path):
        print("Client connected")
        try:
            async for message in websocket:
                # Pad the audio chunk if it's smaller than 1 second
                if len(message) < 32000:
                    padding = bytearray(32000 - len(message))
                    message += padding

                audio_chunk = np.frombuffer(message, dtype=np.int16)
                param_res_list = self.avatar.audio_chunk_to_param(audio_chunk.tobytes())

                for param_res in param_res_list:
                    bg_frame_id = self.frame_index % self.avatar.bg_video_frame_count
                    mouth_img = self.avatar.param2img(param_res, bg_frame_id)
                    full_img, _ = self.avatar.merge_mouth_to_bg(mouth_img, bg_frame_id)
                    
                    # Encode the frame as JPEG and then as base64
                    _, buffer = cv2.imencode('.jpg', full_img)
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                    await websocket.send(jpg_as_text)
                    self.frame_index += 1

        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")

async def main():
    service = LiteAvatarService()
    server = await websockets.serve(service.handle_connection, "localhost", 8765)
    print("WebSocket server started at ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
