import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import shutil
import tempfile
from lite_avatar import liteAvatar
import subprocess as sp
from loguru import logger

app = FastAPI()

def get_avatar_dir():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "resource", "avatar")

def _download_from_modelscope(avatar_name: str) -> str:
    """
    download avatar data from modelscope to resource/avatar/liteavatar
    return avatar_zip_path
    """
    if not avatar_name.endswith(".zip"):
        avatar_name = avatar_name + ".zip"
    avatar_dir = get_avatar_dir()
    if not os.path.exists(avatar_dir):
        os.makedirs(avatar_dir)
    avatar_zip_path = os.path.join(avatar_dir, avatar_name)
    if not os.path.exists(avatar_zip_path):
        # cmd = [
        #     "modelscope", "download", "--model", "HumanAIGC-Engineering/LiteAvatarGallery", avatar_name,
        #     "--local_dir", avatar_dir
        #     ]
        logger.info("download avatar data from modelscope, {}", " ".join("https://modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery/files"))
        # sp.run(cmd)
    return avatar_zip_path

def _get_avatar_data_dir(avatar_name):
    logger.info("use avatar name {}", avatar_name)
    avatar_zip_path = _download_from_modelscope(avatar_name)
    avatar_dir = get_avatar_dir()
    extract_dir = os.path.join(avatar_dir, os.path.dirname(avatar_name))
    avatar_data_dir = os.path.join(avatar_dir, avatar_name)
    if not os.path.exists(avatar_data_dir):
        # extract avatar data to dir
        logger.info("extract avatar data to dir {}", extract_dir)
        assert os.path.exists(avatar_zip_path)
        shutil.unpack_archive(avatar_zip_path, extract_dir)
    assert os.path.exists(avatar_data_dir)
    return avatar_data_dir

@app.post("/generate")
async def generate(audio: UploadFile = File(...), avatar_name: str = Form("sample_data")):
    data_dir = _get_avatar_data_dir(avatar_name)
    lite_avatar = liteAvatar(data_dir=data_dir, num_threads=1, generate_offline=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
        shutil.copyfileobj(audio.file, tmp_audio_file)
        tmp_audio_path = tmp_audio_file.name

    result_dir = tempfile.mkdtemp()

    video_path = lite_avatar.handle(tmp_audio_path, result_dir)

    return FileResponse(video_path, media_type="video/mp4", filename="result.mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)