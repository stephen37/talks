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
    # if os.path.exists(output_folder):
    #     import shutil

    #     shutil.rmtree(output_folder)
    # Path(output_folder).mkdir(parents=True, exist_ok=True)

    local_video_path = video_to_images(video_path, output_folder)
    video_to_audio(local_video_path, output_audio_path)
    text_data = audio_to_text(output_audio_path)
    with open(os.path.join(output_folder, "output_text.txt"), "w") as file:
        file.write(text_data)
    os.remove(output_audio_path)

    return {
        "Author": "3blue1brown",
        "Title": "Backpropagation, step-by-step | DL3",
        "Views": 5067011,
    }


def create_index(output_folder: str) -> MultiModalVectorStoreIndex:
    text_store = MilvusVectorStore(
        uri="http://127.0.0.1:19530",
        collection_name="text_collection",
        overwrite=False,
        dim=1024,
    )
    image_store = MilvusVectorStore(
        uri="http://127.0.0.1:19530",
        collection_name="image_collection",
        overwrite=False,
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


def load_existing_index() -> MultiModalVectorStoreIndex | None:
    """Loads an existing index from Milvus."""
    try:
        st.info("Connecting to Milvus and attempting to load index collections...")
        text_store = MilvusVectorStore(
            uri="http://127.0.0.1:19530",
            collection_name="text_collection",
            overwrite=False,  # Important: Don't overwrite
            dim=1024,
        )
        image_store = MilvusVectorStore(
            uri="http://127.0.0.1:19530",
            collection_name="image_collection",
            overwrite=False,  # Important: Don't overwrite
            dim=512,
        )
        storage_context = StorageContext.from_defaults(
            vector_store=text_store, image_store=image_store
        )
        # Instantiate the index object using the existing storage context
        # Pass an empty list for documents as we are not building from source
        index = MultiModalVectorStoreIndex([], storage_context=storage_context)

        # Perform a simple check to ensure retriever can be created
        _ = index.as_retriever(similarity_top_k=1, image_similarity_top_k=1)
        st.info("Index components loaded successfully.")
        return index
    except Exception as e:
        st.error(f"Error loading existing index: {e}")
        st.warning(
            "Ensure Milvus is running and the collections 'text_collection' and 'image_collection' exist."
        )
        return None


def main():
    st.title("Multimodal RAG with Pixtral & Milvus")

    if "index" not in st.session_state:
        st.session_state.index = None
        st.session_state.retriever_engine = None
        st.session_state.metadata = None # Assuming metadata might be stored/retrieved elsewhere if not processing video

    build_index_from_video = st.checkbox("Build index from new video?", value=False)
    video_path = st.text_input("Enter video path (required if building index):")

    # Logic to build or load index
    if build_index_from_video:
        if video_path:
            if st.button("Process Video and Build Index"):
                output_folder = "./mixed_data/"
                output_audio_path = "./mixed_data/output_audio.wav"

                with st.spinner("Processing video..."):
                    # For demo purposes, let's keep the dummy metadata generation
                    st.session_state.metadata = process_video(
                        video_path, output_folder, output_audio_path
                    )
                with st.spinner("Creating index..."):
                    st.session_state.index = create_index(output_folder)
                    st.session_state.retriever_engine = st.session_state.index.as_retriever(
                        similarity_top_k=5, image_similarity_top_k=5
                    )
                st.success("Video processed and index created!")
        else:
            st.warning("Please enter a video path if you want to build a new index.")
    elif not st.session_state.index: # Attempt to load existing index if not building and index not already loaded
        if st.button("Load Existing Index from Milvus"):
             with st.spinner("Attempting to load existing index..."):
                st.session_state.index = load_existing_index()
                if st.session_state.index:
                     st.session_state.retriever_engine = st.session_state.index.as_retriever(
                        similarity_top_k=5, image_similarity_top_k=5
                    )
                     # You might want to load/set metadata here if it's relevant for queries
                     # For demo, setting dummy metadata if index loaded successfully
                     # Let's use the same dummy data structure as process_video for consistency
                     st.session_state.metadata = {
                         "Author": "Loaded Index",
                         "Title": "Unknown (Loaded)",
                         "Views": 0,
                     }
                     st.success("Existing index loaded!")
                else:
                    st.error("Failed to load existing index. Check Milvus connection and collection status.")


    # Chat interface - only show if index is loaded/created
    if st.session_state.index and st.session_state.retriever_engine:
        st.subheader("Chat with the Video Index")
        query = st.text_input("Ask a question:")

        if query:
            with st.spinner("Generating response..."):
                # Assume metadata is available (loaded or from processing)
                if not st.session_state.metadata:
                     st.warning("Metadata not available for context.")
                     metadata_str = "{}" # Default empty metadata
                else:
                     metadata_str = json.dumps(st.session_state.metadata)

                img_paths, txt_snippets = retrieve(
                    retriever_engine=st.session_state.retriever_engine, query_str=query
                )

                # Handle case where no images are retrieved
                image_document = None
                image_to_display_path = None # Keep track of which image path is used
                if img_paths:
                     # Need to load the actual ImageDocument for processing
                     # Assuming images are stored locally relative to where the script runs
                     # or accessible via the path stored in metadata. Adjust as needed.
                     try:
                        # Use the first retrieved image path.
                        # We assume the path stored in metadata is correct and accessible.
                        # For the demo, it's likely relative like './mixed_data/framexxxx.png'
                        first_img_path = img_paths[0]

                        # Ensure the path exists before trying to load
                        if os.path.exists(first_img_path):
                            image_documents = SimpleDirectoryReader(
                                input_files=[first_img_path]
                             ).load_data()

                            if image_documents:
                                 image_document = image_documents[0]
                                 image_to_display_path = first_img_path # Store path for display
                        else:
                             st.warning(f"Retrieved image path not found: {first_img_path}")

                     except Exception as e:
                         st.error(f"Error loading retrieved image: {e}")


                context_str = "\\n".join(txt_snippets) # Join text snippets for context

                if image_document: # Only call Pixtral if an image was successfully loaded
                    response = process_query_with_image(
                        query, context_str, metadata_str, image_document
                    )
                    st.write("Response:", response)

                    # Display the specific image used for the query
                    st.subheader("Image Context:")
                    st.image(
                        image_to_display_path, # Use the stored path
                        caption=f"Used Image: {os.path.basename(image_to_display_path)}",
                        use_container_width=True,
                    )
                else: # Fallback or indicate no image context was used
                    st.warning("No relevant image found or loaded for this query. Answering based on text context only.")
                    # Optionally, you could have a text-only processing function here
                    # For now, just display the text context
                    st.write("Retrieved Text Context:", context_str)


    elif build_index_from_video and video_path:
        st.info("Click 'Process Video and Build Index' to start.")
    elif not build_index_from_video:
         st.info("Click 'Load Existing Index from Milvus' or check the box above to build a new one.")
    # Removed the redundant 'else' block which could cause confusion.
    # The default state (no option clicked yet) is covered by the st.info messages above.


if __name__ == "__main__":
    main()
