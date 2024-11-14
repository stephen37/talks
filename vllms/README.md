# MultiModal RAG with Pixtral & Milvus

This project is to showcase an application that processes videos to extract images and audio, transcribes the audio to text, and creates a multi-modal index using Milvus. It allows users to interact with the video content by asking questions.

## Features

- **Video Processing**: Extracts frames and audio from a video file.
- **Audio Transcription**: Converts audio to text using speech recognition.
- **Index Creation**: Builds a multi-modal index with text and image data using Milvus.
- **Query Processing**: Allows users to ask questions about the video content and retrieves relevant information using Pixtral.
- **Image Display**: Shows relevant images used in the response.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/stephen37/talks/vllms.git
   cd vllms
   ```

2. Install the required packages:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your API keys and endpoints:
     ```
     MISTRAL_API_KEY=your_mistral_api_key
     KOYEB_ENDPOINT=your_koyeb_endpoint
     KOYEB_TOKEN=your_koyeb_token
     ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run multimodal_blogpost_rag.py
   ```

2. Enter the path to a video file in the input field.

3. Wait for the video to be processed and the index to be created.

4. Ask questions about the video content in the provided input field.

5. View the response and relevant images displayed by the application.
