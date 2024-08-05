import gradio as gr
import torch
import numpy as np
from diffusers import I2VGenXLPipeline
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from PIL import Image
from moviepy.editor import ImageSequenceClip
import io
import scipy.io.wavfile
import ffmpeg

def generate_video(image, prompt, negative_prompt, video_length):
    generator = torch.manual_seed(8888)

    # Set the device to CPU or a non-NVIDIA GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pipeline
    pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float32)
    pipeline.to(device)  # Move the model to the selected device

    # Generate frames with progress tracking
    frames = []
    total_frames = video_length * 20  # Assuming 30 frames per second

    for i in range(total_frames):
        frame = pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=1,
            negative_prompt=negative_prompt,
            guidance_scale=9.0,
            generator=generator,
            num_frames=1
        ).frames[0]
        frames.append(np.array(frame))

        # Update progress
        yield (i + 1) / total_frames  # Yield progress

    # Create a video clip from the frames
    output_file = "output_video.mp4"
    clip = ImageSequenceClip(frames, fps=30)  # Set the frames per second
    clip.write_videofile(output_file, codec='libx264', audio=False)

    return output_file

def generate_music(prompt, unconditional=False):
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Generate music
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
    audio_file = "musicgen_out.wav"
    
    # Ensure audio_values is 1D and scale if necessary
    audio_data = audio_values[0].cpu().numpy()
    
    # Check if audio_data is in the correct format
    if audio_data.ndim > 1:
        audio_data = audio_data[0]  # Take the first channel if stereo

    # Scale audio data to 16-bit PCM format
    audio_data = np.clip(audio_data, -1.0, 1.0)  # Ensure values are in the range [-1, 1]
    audio_data = (audio_data * 32767).astype(np.int16)  # Scale to int16

    # Save the generated audio
    scipy.io.wavfile.write(audio_file, sampling_rate, audio_data)
    
    return audio_file

def combine_audio_video(audio_file, video_file):
    output_file = "combined_output.mp4"
    audio = ffmpeg.input(audio_file)
    video = ffmpeg.input(video_file)
    output = ffmpeg.output(video, audio, output_file, vcodec='copy', acodec='aac')
    ffmpeg.run(output)
    return output_file

def interface(image_path, prompt, negative_prompt, video_length, music_prompt, unconditional):
    image = Image.open(image_path)
    video_file = generate_video(image, prompt, negative_prompt, video_length)
    audio_file = generate_music(music_prompt, unconditional)
    combined_file = combine_audio_video(audio_file, video_file)
    return combined_file

with gr.Blocks() as demo:
    gr.Markdown("# AI-Powered Video and Music Generation")
    
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload Image")
        prompt_input = gr.Textbox(label="Enter the Video Prompt")
        negative_prompt_input = gr.Textbox(label="Enter the Negative Prompt")
        video_length_input = gr.Number(label="Video Length (seconds)", value=10, precision=0)
        music_prompt_input = gr.Textbox(label="Enter the Music Prompt")
        unconditional_checkbox = gr.Checkbox(label="Generate Unconditional Music")

    generate_button = gr.Button("Generate Video and Music")
    output_video = gr.Video(label="Output Video with Sound")

    generate_button.click(
        interface,
        inputs=[image_input, prompt_input, negative_prompt_input, video_length_input, music_prompt_input, unconditional_checkbox],
        outputs=output_video,
        show_progress=True
    )

demo.launch()
