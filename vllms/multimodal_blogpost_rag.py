import base64
import json
import os
from pathlib import Path

import speech_recognition as sr
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageNode
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from moviepy.editor import VideoFileClip
from openai import OpenAI

load_dotenv()
Settings.embed_model = MistralAIEmbedding(
    "mistral-embed", api_key=os.getenv("MISTRAL_API_KEY")
)


def video_to_images(video_path: str, output_folder: str) -> str:
    local_video_path = video_path
    if video_path.startswith(("http://", "https://")):
        import yt_dlp

        ydl_opts = {
            "format": "best",
            "outtmpl": os.path.join(output_folder, "video.mp4"),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_path])
        local_video_path = os.path.join(output_folder, "video.mp4")

    clip = VideoFileClip(local_video_path)
    clip.write_images_sequence(os.path.join(output_folder, "frame%04d.png"), fps=0.2)
    return local_video_path


def video_to_audio(video_path: str, output_audio_path: str) -> None:
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)


def audio_to_text(audio_path: str) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_whisper(audio_data)
    return text


def process_video(video_path: str, output_folder: str, output_audio_path: str) -> dict:
    # Clean up existing files
    if os.path.exists(output_folder):
        import shutil

        shutil.rmtree(output_folder)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    local_video_path = video_to_images(video_path, output_folder)
    video_to_audio(local_video_path, output_audio_path)
    text_data = audio_to_text(output_audio_path)
    with open(os.path.join(output_folder, "output_text.txt"), "w") as file:
        file.write(text_data)
    os.remove(output_audio_path)
    return {"Author": "Example Author", "Title": "Example Title", "Views": "1000000"}


def create_index(output_folder: str) -> MultiModalVectorStoreIndex:
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
    return MultiModalVectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )


def retrieve(retriever_engine, query_str) -> tuple[list[str], list[str]]:
    retrieval_results = retriever_engine.retrieve(query_str)
    retrieved_image = [
        res_node.node.metadata["file_path"]
        for res_node in retrieval_results
        if isinstance(res_node.node, ImageNode)
    ]
    retrieved_text = [
        res_node.text
        for res_node in retrieval_results
        if not isinstance(res_node.node, ImageNode)
    ]
    return retrieved_image, retrieved_text


def process_query_with_image(
    query_str, context_str, metadata_str, image_document
) -> str | None:
    client = OpenAI(
        base_url=os.getenv("KOYEB_ENDPOINT"), api_key=os.getenv("KOYEB_TOKEN")
    )
    with open(image_document.image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    qa_tmpl_str = "Given the provided information, including relevant images and retrieved context from the video, accurately and precisely answer the query without any additional prior knowledge.\nPlease ensure honesty and responsibility.\n---------------------\nContext: {context_str}\nMetadata for video: {metadata_str} \n---------------------\nQuery: {query_str}\nAnswer: "

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

    completion = client.chat.completions.create(
        model="mistralai/Pixtral-12B-2409", messages=messages, max_tokens=1000
    )
    return completion.choices[0].message.content


def main():
    st.title("Multimodal RAG with Pixtral & Milvus")

    if "index" not in st.session_state:
        st.session_state.index = None
        st.session_state.retriever_engine = None
        st.session_state.metadata = None

    video_path = st.text_input("Enter video path:")

    if video_path and not st.session_state.index:
        output_folder = "./mixed_data/"
        output_audio_path = "./mixed_data/output_audio.wav"

        with st.spinner("Processing video..."):
            st.session_state.metadata = process_video(
                video_path, output_folder, output_audio_path
            )

        with st.spinner("Creating index..."):
            st.session_state.index = create_index(output_folder)
            st.session_state.retriever_engine = st.session_state.index.as_retriever(
                similarity_top_k=5, image_similarity_top_k=5
            )

        st.success("Video processed and index created!")

    if st.session_state.index:
        st.subheader("Chat with the Video")
        query = st.text_input("Ask a question about the video:")

        if query:
            with st.spinner("Generating response..."):
                img, txt = retrieve(
                    retriever_engine=st.session_state.retriever_engine, query_str=query
                )
                image_documents = SimpleDirectoryReader(
                    input_dir="./mixed_data/", input_files=img
                ).load_data()
                context_str = "".join(txt)
                metadata_str = json.dumps(st.session_state.metadata)

                response = process_query_with_image(
                    query, context_str, metadata_str, image_documents[0]
                )
                st.write("Response:", response)

                # Display the images used in the explanation
                st.subheader("Relevant Images:")
                cols = st.columns(
                    min(3, len(image_documents))
                )  # Create up to 3 columns
                for idx, image_doc in enumerate(image_documents):
                    with cols[idx % 3]:
                        st.image(
                            image_doc.image_path,
                            caption=f"Image {idx + 1}",
                            use_column_width=True,
                        )
    elif video_path:
        st.info("Processing video and creating index. Please wait...")
    else:
        st.info("Please enter a video path to begin.")


if __name__ == "__main__":
    main()
