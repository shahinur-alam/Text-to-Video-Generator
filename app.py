import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from flask import Flask, render_template, request, send_file
from diffusers import DiffusionPipeline
import uuid
import tempfile
import traceback
import numpy as np
import imageio
import cv2

app = Flask(__name__)

# Check CUDA availability
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# Initialize the text-to-video pipeline
try:
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16", force_download=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    print(f"Pipeline loaded successfully and moved to {device}")
except Exception as e:
    print("Error initializing pipeline:", str(e))
    print(traceback.format_exc())

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        num_frames = int(request.form['num_frames'])

        try:
            # Generate video
            video_path = generate_video(prompt, num_frames)
            return render_template('index.html', video_path=video_path)
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            print(traceback.format_exc())
            return render_template('index.html', error=error_message)

    return render_template('index.html')


def generate_video(prompt, num_frames):
    print(f"Generating video for prompt: '{prompt}' with {num_frames} frames")
    try:
        # Generate video frames
        video_frames = pipe(prompt, num_inference_steps=30, num_frames=num_frames).frames
        print(f"Generated {len(video_frames)} frames")
        print(f"First frame shape: {video_frames[0].shape}")

        # Process frames
        processed_frames = []
        for frame in video_frames:
            if frame.shape[0] == 16:  # If the first dimension is 16
                frame = frame[0]  # Take the first frame from the batch
            frame = (frame * 255).astype(np.uint8)
            processed_frames.append(frame)

        print(f"Processed frame shape: {processed_frames[0].shape}")

        # Export frames to video
        video_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")

        # Use OpenCV to save the video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 8, (processed_frames[0].shape[1], processed_frames[0].shape[0]))
        for frame in processed_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        print(f"Video exported to: {video_path}")

        return video_path
    except Exception as e:
        print("Error in generate_video:", str(e))
        print(traceback.format_exc())
        raise

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        print("Error in download_file:", str(e))
        print(traceback.format_exc())
        return "Error downloading file", 500

if __name__ == '__main__':
    app.run(debug=True)
