# Text to Video Generator

## Overview

This project is a Flask web application that generates short videos from text prompts using a pre-trained model. The app supports both CPU and GPU execution, detecting CUDA automatically for GPU acceleration.

## Features

- Generate videos from text using the `damo-vilab/text-to-video-ms-1.7b` model.
- Web interface to input prompts and specify the number of frames.
- Preview or download the generated video.
- Supports GPU acceleration (if available).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shahinur-alam/Text-to-Video-Generator.git
   cd Text-to-Video-Generator
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
3. Start the Flask server:

   ```bash
   python app.py
4. Open the app in your browser at http://127.0.0.1:5000.

## Usage

1. Enter a prompt and number of frames in the form.
2. Click "Generate Video."
3. Preview or download the generated .mp4 video.

## API Endpoints

- GET /: Displays the video generation form.
- POST /: Processes the video generation request.
- GET /download/<filename>: Downloads the generated video.

## File Structure

- app.py: Main application.
- templates/index.html: Web interface.
- requirements.txt: Dependencies.