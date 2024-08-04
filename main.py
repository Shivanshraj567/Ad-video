import torch
import skvideo.io
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_video, load_image
import os
import io
from PIL import Image
import numpy as np
import imageio
from moviepy.editor import ImageSequenceClip
import streamlit as st
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from scipy.io import wavfile
from datasets import load_dataset
import ffmpeg

def generate_video(image, prompt, negative_prompt, video_length):
    generator = torch.manual_seed(8888)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float32)
    pipeline.to(device)
    frames = pipeline(
        prompt=prompt,
        image=image,
        num_inference_steps=2,
        negative_prompt=negative_prompt,
        guidance_scale=9.0,
        generator=generator,
        num_frames=video_length*20
    ).frames[0]
    return frames

def export_frames_to_video(frames, output_file):
    frames_np = [np.array(frame) for frame in frames]
    clip = ImageSequenceClip(frames_np, fps=30)
    clip.write_videofile(output_file, codec='libx264', audio=False)

def generate_music(prompt, unconditional=False):
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
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

def main():
    st.title("Media Generation App")
    tabs = st.tabs(["Video Generation", "Music Generation", "Combine Audio and Video"])
    with tabs[0]:
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_data = uploaded_file.read()
            image = Image.open(io.BytesIO(image_data))
            prompt = st.text_input("Enter the prompt:")
            negative_prompt = st.text_input("Enter the negative prompt:")
            video_length = st.number_input("Enter the length of the video in seconds:", min_value=1)
            if st.button("Generate Video"):
                frames = generate_video(image, prompt, negative_prompt, video_length)
                output_file = "output_video.mp4"
                export_frames_to_video(frames, output_file)
                st.success(f"Video saved as {output_file}")
                st.video(output_file)
    with tabs[1]:
        prompt = st.text_input("Enter the music prompt:")
        unconditional = st.checkbox("Generate unconditional music")
        if st.button("Generate Music"):
            audio_values, sampling_rate = generate_music(prompt, unconditional)
            wavfile.write("musicgen_out.wav", sampling_rate, audio_values)
            st.success("Music saved as musicgen_out.wav")
            st.audio("musicgen_out.wav")
    with tabs[2]:
        audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
        video_file = st.file_uploader("Choose a video file", type=["mp4"])
        if audio_file and video_file:
            audio_filename = audio_file.name
            video_filename = video_file.name
            with open(audio_filename, "wb") as f:
                f.write(audio_file.read())
            with open(video_filename, "wb") as f:
                f.write(video_file.read())
            output_file = "combined_output.mp4"
            combine_audio_video(audio_filename, video_filename, output_file)
            st.success(f"Combined audio and video saved as {output_file}")
            st.video(output_file)

if __name__ == "__main__":
    main()
