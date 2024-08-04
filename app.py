import streamlit as st
from PIL import Image
import torch
import skvideo.io
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_video, load_image
import numpy as np
import imageio
from moviepy.editor import ImageSequenceClip
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from scipy.io import wavfile
import ffmpeg
import time

def generate_video(image, prompt, negative_prompt, video_length):
    generator = torch.manual_seed(8888)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float32)
    pipeline.to(device)
    
    # Simulate progress for video generation
    frames = []
    for i in range(video_length * 20):  # Assuming 20 frames per second
        frame = pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=2,
            negative_prompt=negative_prompt,
            guidance_scale=9.0,
            generator=generator,
            num_frames=1
        ).frames[0]
        frames.append(frame)
        st.progress((i + 1) / (video_length * 20))  # Update progress bar
    
    return frames

def export_frames_to_video(frames, output_file):
    frames_np = [np.array(frame) for frame in frames]
    clip = ImageSequenceClip(frames_np, fps=30)
    clip.write_videofile(output_file, codec='libx264', audio=False)

def generate_music(prompt, unconditional=False):
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Simulate progress for music generation
    if unconditional:
        unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
        audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)
    else:
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        inputs = processor(
            text=prompt,
            padding=True,
            return_tensors="pt",
        )
        audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=256)
    
    sampling_rate = model.config.audio_encoder.sampling_rate
    return audio_values[0].cpu().numpy(), sampling_rate

def combine_audio_video(audio_file, video_file, output_file):
    audio = ffmpeg.input(audio_file)
    video = ffmpeg.input(video_file)
    output = ffmpeg.output(video, audio, output_file, vcodec='copy', acodec='aac')
    ffmpeg.run(output)

st.title("AI-Powered Video and Music Generation")

st.sidebar.title("Options")

st.sidebar.subheader("Video Generation")
image = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
prompt = st.sidebar.text_input("Enter the prompt")
negative_prompt = st.sidebar.text_input("Enter the negative prompt")
video_length = st.sidebar.number_input("Enter the video length (seconds)", min_value=1, value=10)

st.sidebar.subheader("Music Generation")
music_prompt = st.sidebar.text_input("Enter the music prompt")
unconditional = st.sidebar.checkbox("Generate unconditional music")

if st.sidebar.button("Generate Video and Music"):
    if image is not None:
        image = Image.open(image)
        
        # Video generation with progress bar
        st.write("Generating video...")
        video_frames = generate_video(image, prompt, negative_prompt, video_length)
        export_frames_to_video(video_frames, "output_video.mp4")
        st.video("output_video.mp4")

    # Music generation with progress bar
    st.write("Generating music...")
    audio_values, sampling_rate = generate_music(music_prompt, unconditional)
    wavfile.write("musicgen_out.wav", sampling_rate, audio_values)
    st.audio("musicgen_out.wav")

    # Combine audio and video
    st.write("Combining audio and video...")
    combine_audio_video("musicgen_out.wav", "output_video.mp4", "combined_output.mp4")
    st.video("combined_output.mp4")
    
