import asyncio
import websockets
import json
from lite_avatar import liteAvatar
import numpy as np

class LiteAvatarService:
    def __init__(self):
        # self.avatar = liteAvatar(data_dir='D:/pei2/lite-avatar/data/sample_data', num_threads=1, generate_offline=True)
        self.avatar = liteAvatar(data_dir='D:/pei2/lite-avatar/resource/avatar', num_threads=1, generate_offline=True)
        self.audio_buffer = bytearray()

    async def handle_connection(self, websocket, path):
        print("Client connected")
        try:
            async for message in websocket:
                # Pad the audio chunk if it's smaller than 1 second
                if len(message) < 32000:
                    padding = bytearray(32000 - len(message))
                    message += padding

                audio_chunk = np.frombuffer(message, dtype=np.int16)
                param_res = self.avatar.audio_chunk_to_param(audio_chunk.tobytes())
                
                # In a real implementation, you would send the param_res back to the client
                # For now, we'll just print it
                print("Generated animation data")

                await websocket.send(json.dumps(param_res))

        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")

async def main():
    service = LiteAvatarService()
    server = await websockets.serve(service.handle_connection, "localhost", 8765)
    print("WebSocket server started at ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
