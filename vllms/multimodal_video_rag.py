import os
import base64
import json
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import requests
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from pytube import YouTube
from openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.schema import ImageNode
import streamlit as st


load_dotenv()

Settings.embed_model = MistralAIEmbedding(
    "mistral-embed", api_key=os.getenv("MISTRAL_API_KEY")
)

def download_video(url: str, output_path: str) -> Dict[str, str]:
    try:
        yt = YouTube(url)
        metadata = {"Author": yt.author, "Title": yt.title, "Views": str(yt.views)}
        yt.streams.get_highest_resolution().download(
            output_path=output_path, filename="input_vid.mp4"
        )
    except Exception as e:
        print(f"Error with pytube: {e}")
        print("Attempting direct download...")

        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(output_path, "input_vid.mp4"), "wb") as f:
                f.write(response.content)
            metadata = {"Author": "Unknown", "Title": "Unknown", "Views": "Unknown"}
        else:
            raise Exception("Failed to download video directly.")

    return metadata

def video_to_images(video_path: str, output_folder: str) -> None:
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(os.path.join(output_folder, "frame%04d.png"), fps=0.2)

def video_to_audio(video_path: str, output_audio_path: str) -> None:
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)

def audio_to_text(audio_path: str) -> str:
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)

    with audio as source:
        audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
            text = ""
        except sr.RequestError as e:
            print(f"Could not request results from service; {e}")
            text = ""

    return text

def process_video(video_url: str, output_video_path: str, output_folder: str, output_audio_path: str) -> Dict[str, str]:
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_video_path, "gaussian.mp4")
    
    # metadata = download_video(video_url, output_video_path)

    metadata = {
        "Author": "3Blue1Brown",
        "Title": "A pretty reason why Gaussian + Gaussian = Gaussian",
        "Views": 803400,
    }
    video_to_images(filepath, output_folder)
    video_to_audio(filepath, output_audio_path)
    text_data = audio_to_text(output_audio_path)

    with open(os.path.join(output_folder, "output_text.txt"), "w") as file:
        file.write(text_data)
    print("Text data saved to file")

    os.remove(output_audio_path)
    print("Audio file removed")

    return metadata

def create_index(output_folder: str):
    text_store = MilvusVectorStore(
        uri="http://127.0.0.1:19530",
        collection_name="text_collection",
        overwrite=True,
        dim=1024,
    )
    image_store = MilvusVectorStore(
        uri="http://127.0.0.1:19530",
        collection_name="image_collection",
        overwrite=True,
        dim=512,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    documents = SimpleDirectoryReader(output_folder).load_data()

    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    return index

def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text

def process_query_with_single_image(query_str, context_str, metadata_str, image_document):
    print("Calling Pixtral for single image processing")
    client = OpenAI(
        base_url=os.getenv("KOYEB_ENDPOINT"),
        api_key=os.getenv("KOYEB_TOKEN"),
    )

    with open(image_document.image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    qa_tmpl_str = (
        "Given the provided information, including relevant images and retrieved context from the video, \
     accurately and precisely answer the query without any additional prior knowledge.\n"
        "Please ensure honesty and responsibility."
        "---------------------\n"
        "Context: {context_str}\n"
        "Metadata for video: {metadata_str} \n"
        "---------------------\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": qa_tmpl_str.format(
                        context_str=context_str,
                        query_str=query_str,
                        metadata_str=metadata_str,
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="mistralai/Pixtral-12B-2409", messages=messages, max_tokens=300
        )
        response = completion.choices[0].message.content
        print(f"Pixtral response for image {image_document.image_path}: {response}")
        return response
    except Exception as e:
        return f"Error processing image {image_document.image_path}: {str(e)}"

def process_query_with_multiple_images(query_str, context_str, metadata_str, image_documents):
    responses = []
    for img_doc in image_documents:
        response = process_query_with_single_image(
            query_str, context_str, metadata_str, img_doc
        )
        responses.append(response)

    combined_response = "\n\n".join(responses)
    return combined_response


def main():
    st.title("MultiModal RAG with Pixtral & Milvus")

    # Use session state to store persistent data
    if 'index' not in st.session_state:
        st.session_state.index = None
        st.session_state.retriever_engine = None
        st.session_state.metadata = None

    # Input for YouTube URL
    video_url = st.text_input("Enter YouTube URL:")

    if video_url and not st.session_state.index:
        video_url = "https://www.youtube.com/watch?v=d_qvLDhkg00"
        output_video_path = "./video_data/"
        output_folder = "./mixed_data/"
        output_audio_path = "./mixed_data/output_audio.wav"

        # Process video
        with st.spinner("Processing video..."):
            st.session_state.metadata = process_video(video_url, output_video_path, output_folder, output_audio_path)

        # Display video preview
        st.video(video_url)

        # Create index
        with st.spinner("Creating index..."):
            st.session_state.index = create_index(output_folder)
            st.session_state.retriever_engine = st.session_state.index.as_retriever(similarity_top_k=5, image_similarity_top_k=5)
        
        st.success("Video processed and index created!")

    # Chat interface
    if st.session_state.index:
        st.subheader("Chat with the Video")
        query = st.text_input("Ask a question about the video:")

        if query:
            with st.spinner("Generating response..."):
                img, txt = retrieve(retriever_engine=st.session_state.retriever_engine, query_str=query)
                image_documents = SimpleDirectoryReader(
                    input_dir="./mixed_data/", input_files=img
                ).load_data()
                context_str = "".join(txt)
                metadata_str = json.dumps(st.session_state.metadata)

                response = process_query_with_multiple_images(
                    query, context_str, metadata_str, image_documents
                )
                st.write("Response:", response)
    elif video_url:
        st.info("Processing video and creating index. Please wait...")
    else:
        st.info("Please enter a YouTube URL to begin.")

if __name__ == "__main__":
    main()