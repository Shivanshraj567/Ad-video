from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os
import torch
import skvideo.io
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_video, load_image
import numpy as np
import imageio
from moviepy.editor import ImageSequenceClip
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from scipy.io import wavfile
from datasets import load_dataset
import ffmpeg

app = Flask(__name__)

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

@app.route('/generate_video', methods=['POST'])
def generate_video_route():
    image = request.files['image']
    prompt = request.form['prompt']
    negative_prompt = request.form['negative_prompt']
    video_length = int(request.form['video_length'])
    frames = generate_video(image, prompt, negative_prompt, video_length)
    output_file = "output_video.mp4"
    export_frames_to_video(frames, output_file)
    return send_file(output_file, as_attachment=True)

@app.route('/generate_music', methods=['POST'])
def generate_music_route():
    prompt = request.form['prompt']
    unconditional = request.form['unconditional'] == 'true'
    audio_values, sampling_rate = generate_music(prompt, unconditional)
    wavfile.write("musicgen_out.wav", sampling_rate, audio_values)
    return send_file("musicgen_out.wav", as_attachment=True)

@app.route('/combine_audio_video', methods=['POST'])
def combine_audio_video_route():
    audio_file = request.files['audio_file']
    video_file = request.files['video_file']
    output_file = "combined_output.mp4"
    combine_audio_video(audio_file, video_file, output_file)
    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True,port=5000)
