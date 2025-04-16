import streamlit as st
st.set_page_config(page_title="Live AI Narrator", layout="wide")

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import pyttsx3
import threading
import multiprocessing

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Load BLIP model
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Shared state
caption_container = st.empty()

# Video processor
class BLIPCaptioner(VideoProcessorBase):
    def __init__(self):
        self.caption = "Waiting for input..."
        self.processor, self.model = load_blip_model()
        self.last_caption = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        pil_image = Image.fromarray(img[..., ::-1])  # Convert BGR to RGB

        inputs = self.processor(images=pil_image, return_tensors="pt")
        output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)

        # Update if caption changed
        if caption != self.last_caption:
            self.caption = caption
            self.last_caption = caption
            caption_container.markdown(f"### 🧠 Caption: `{caption}`")
            threading.Thread(target=self.speak_caption, args=(caption,), daemon=True).start()

        return frame

    def speak_caption(self, text):
        tts_engine.say(text)
        tts_engine.runAndWait()

# Streamlit App
def main():
    st.title("🎥 AI Video Narrator with BLIP (Caption + Voice)")

    webrtc_streamer(
        key="live-caption",
        video_processor_factory=BLIPCaptioner,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# Required for Windows multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
