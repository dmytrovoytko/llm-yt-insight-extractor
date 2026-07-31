"""YT Insight Extractor - Main Streamlit Application

Provides a web interface for extracting actionable insights from YouTube videos
using RAG-powered processing pipeline.
"""

import streamlit as st

from core.transcript import extract_video_id, fetch_youtube_transcript

# Page config
st.set_page_config(
    page_title="YT Insight Extractor",
    page_icon="🎬",
    layout="wide",
)

AVAILABLE_AREAS = [
    "Career",
    "Fitness",
    "Finance",
    "Relationships",
    "Mental Health",
    "Productivity",
    "Learning",
]

SESSION_STATE_KEYS = [
    "pipeline_step",
    "transcript_data",
    "chunks",
    "rag_engine",
    "retrieved_context",
    "final_output",
    "error_message",
]


def initialize_session_state() -> None:
    """Initialize session state variables for pipeline state management."""
    for key in SESSION_STATE_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def run_pipeline_step_1_extract_transcript(
    youtube_url: str,
) -> tuple[list[dict], str]:
    """Step 1: Extract and validate transcript.

    Args:
        youtube_url: YouTube video URL.

    Returns:
        Tuple of (transcript_data, status_message) on success.

    Raises:
        Exception: With error message if extraction fails.
    """
    # Extract video ID
    try:
        video_id = extract_video_id(youtube_url)
    except ValueError as e:
        raise Exception(f"Invalid YouTube URL: {str(e)}")

    # Fetch transcript
    transcript_data = fetch_youtube_transcript(youtube_url)

    # Calculate duration and word count
    if transcript_data:
        total_duration = (
            float(transcript_data[-1]["start"])
            + float(transcript_data[-1]["duration"])
            - float(transcript_data[0]["start"])
        )
        total_words = sum(
            len(entry["text"].split()) for entry in transcript_data
        )
        minutes = int(total_duration) // 60
        seconds = int(total_duration) % 60

        status_msg = f"Transcript extracted, {minutes}min {seconds}s, {total_words} words"
        return transcript_data, status_msg
    else:
        raise Exception(
            "Could not retrieve transcript for this video. It may not have subtitles available."
        )


def run_pipeline_step_2_chunk_and_vectorize(
    transcript_data: list[dict],
) -> tuple:
    """Step 2: Chunk transcript and add to vector store.

    Args:
        transcript_data: List of transcript entries.

    Returns:
        Tuple of (chunks, rag_engine) on success.

    Raises:
        Exception: If chunking or vectorization fails.
    """
    from core.chunker import chunk_transcript
    from core.rag_engine import create_rag_engine

    # Chunk the transcript
    chunks = chunk_transcript(transcript_data)

    if not chunks:
        raise Exception("Failed to chunk transcript.")

    # Create RAG engine and add chunks
    rag_engine = create_rag_engine()
    num_chunks_added = rag_engine.add_chunks(chunks)

    if num_chunks_added == 0:
        raise Exception("Failed to vectorize chunks.")

    return chunks, rag_engine


def display_pipeline_progress(
    youtube_url: str, area_of_life: str, specific_goal: str
):
    """Execute and display pipeline progress for steps 1-2.

    Args:
        youtube_url: YouTube video URL.
        area_of_life: Selected area of life.
        specific_goal: Optional specific goal.
    """
    st.markdown("---")
    st.subheader("⚙️ Processing Pipeline")

    # Step 1: Extract Transcript
    with st.status(
        "🔄 Step 1: Extracting transcript...", expanded=True
    ) as step1_status:
        try:
            transcript_data, status_msg = (
                run_pipeline_step_1_extract_transcript(youtube_url)
            )
            st.session_state.transcript_data = transcript_data
            st.session_state.pipeline_step = "step_1_complete"

            step1_status.update(
                label=f"✅ Step 1: {status_msg}", state="complete"
            )

        except Exception as e:
            st.session_state.error_message = str(e)
            st.session_state.pipeline_step = "error"
            step1_status.update(label=f"❌ Step 1: {str(e)}", state="error")
            st.error(str(e))
            return

    # Step 2: Chunk & Vectorize
    with st.status(
        "🔄 Step 2: Chunking text & Vectorizing...", expanded=True
    ) as step2_status:
        try:
            chunks, rag_engine = run_pipeline_step_2_chunk_and_vectorize(
                st.session_state.transcript_data
            )
            st.session_state.chunks = chunks
            st.session_state.rag_engine = rag_engine
            st.session_state.pipeline_step = "step_2_complete"

            step2_status.update(
                label=f"✅ Step 2: Created {len(chunks)} chunks",
                state="complete",
            )

            # Display chunk summary
            st.success(
                f"✨ Successfully created {len(chunks)} chunks with embeddings"
            )

        except Exception as e:
            st.session_state.error_message = str(e)
            st.session_state.pipeline_step = "error"
            step2_status.update(label=f"❌ Step 2: {str(e)}", state="error")
            st.error(str(e))
            return

    # Display success state
    st.markdown("---")
    st.info(
        "✅ Transcript extraction and vectorization complete. Step 3 (RAG & LLM) coming soon!"
    )


def main() -> None:
    """Main Streamlit application entry point."""
    initialize_session_state()
    st.title("🎬 YT Insight Extractor")
    st.markdown(
        "Provide a YouTube video link, choose an area of life, and optionally add a goal to generate targeted insights."
    )

    with st.form(key="input_form"):
        youtube_url = st.text_input(
            "YouTube URL", placeholder="https://www.youtube.com/watch?v=...",
            value="https://www.youtube.com/watch?v=TrvLEgPpV8s" # Initial Value
        )
        area_of_life = st.selectbox("Area of Life", AVAILABLE_AREAS)
        specific_goal = st.text_input(
            "Your specific goal (optional)",
            placeholder="E.g. improve focus, build better habits, learn negotiation",
        )
        submit_button = st.form_submit_button("🚀 Process")

    if submit_button:
        if not youtube_url:
            st.error("Please enter a YouTube URL before processing.")
            return

        st.session_state.pipeline_step = "inputs_received"
        st.session_state.transcript_data = None
        st.session_state.chunks = None
        st.session_state.retrieved_context = None
        st.session_state.final_output = None
        st.session_state.error_message = None

        # st.markdown("---")
        # st.subheader("📝 Received inputs")
        # st.write("**YouTube URL:**", youtube_url)
        # st.write("**Area of Life:**", area_of_life)
        # st.write("**Specific Goal:**", specific_goal or "(none)")

        # Run the pipeline for steps 1-2
        display_pipeline_progress(youtube_url, area_of_life, specific_goal)


if __name__ == "__main__":
    main()
