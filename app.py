"""YT Insight Extractor - Main Streamlit Application

Provides a web interface for extracting actionable insights from YouTube videos
using RAG-powered processing pipeline.
"""

import streamlit as st

from core.exports import (
    convert_timestamps_to_youtube_links,
    export_actionable_ideas_to_markdown,
    export_subtopics_to_markdown,
    export_to_json,
    export_to_markdown,
)
from core.prompts import ActionableIdeas, Subtopics
from core.transcript import extract_video_id, fetch_youtube_transcript
from core.history import get_history_store, load_history

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


def build_rag_query(area_of_life: str, specific_goal: str) -> str:
    """Create a simple retrieval query for RAG.

    Args:
        area_of_life: Selected area of life.
        specific_goal: Optional specific goal.

    Returns:
        A combined query string.
    """
    query = f"Extract insights about {area_of_life}."
    if specific_goal:
        query += f" The user wants to: {specific_goal}."
    return query


def run_pipeline_step_3_rag_and_summarize(
    area_of_life: str,
    specific_goal: str,
    rag_engine,
    top_k: int = 5,
) -> tuple[list[dict], str]:
    """Step 3: Retrieve relevant transcript context and summarize.

    Args:
        area_of_life: Selected area of life.
        specific_goal: Optional specific goal.
        rag_engine: Initialized RAG engine.
        top_k: Number of top chunks to retrieve.

    Returns:
        Tuple of (retrieved_context, status_message).

    Raises:
        Exception: If retrieval fails.
    """
    query = build_rag_query(area_of_life, specific_goal)
    results = rag_engine.retrieve(query, top_k=top_k)

    if not results:
        raise Exception(
            "No relevant context could be retrieved from the transcript."
        )

    status_msg = (
        f"Retrieved {len(results)} relevant transcript chunks "
        f"for RAG & summarization."
    )
    return results, status_msg


def display_pipeline_progress(
    youtube_url: str, area_of_life: str, specific_goal: str
):
    """Execute and display pipeline progress for steps 1-3.

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

    # Step 3: RAG & Summarization
    with st.status(
        "🔄 Step 3: RAG & Summarization...", expanded=True
    ) as step3_status:
        try:
            retrieved_context, status_msg = (
                run_pipeline_step_3_rag_and_summarize(
                    area_of_life,
                    specific_goal,
                    st.session_state.rag_engine,
                )
            )
            st.session_state.retrieved_context = retrieved_context
            st.session_state.final_output = {
                "query": build_rag_query(area_of_life, specific_goal),
                "retrieved_context": retrieved_context,
            }
            st.session_state.pipeline_step = "step_3_complete"

            step3_status.update(
                label=f"✅ Step 3: {status_msg}", state="complete"
            )
            st.success(
                f"🚀 Step 3 complete: {len(retrieved_context)} chunks retrieved for summarization."
            )

        except Exception as e:
            st.session_state.error_message = str(e)
            st.session_state.pipeline_step = "error"
            step3_status.update(label=f"❌ Step 3: {str(e)}", state="error")
            st.error(str(e))
            return

    st.markdown("---")
    st.info(
        "✅ Pipeline complete. Review retrieved context below and continue to build the LLM summary flow."
    )


def render_results_tabs(
    subtopics: Subtopics,
    actionable_ideas: ActionableIdeas,
    video_id: str,
) -> None:
    """Render extracted results in two tabs with clickable timestamps.

    Args:
        subtopics: Extracted subtopics Pydantic model
        actionable_ideas: Extracted actionable ideas Pydantic model
        video_id: YouTube video ID for creating timestamp links
    """
    tab1, tab2 = st.tabs(["📚 Subtopics Summary", "💡 Top 5 Actionable Ideas"])

    with tab1:
        st.markdown("## Subtopics Summary\n")
        for idx, subtopic in enumerate(subtopics.subtopics, start=1):
            # Convert timestamp to clickable link
            timestamp_with_link = convert_timestamps_to_youtube_links(
                subtopic.timestamp, video_id
            )

            with st.container(border=True):
                st.subheader(subtopic.title)
                st.markdown(f"**Timestamp:** {timestamp_with_link}")
                st.write(subtopic.summary)

    with tab2:
        st.markdown("## Top 5 Actionable Ideas\n")
        for idx, idea in enumerate(actionable_ideas.ideas, start=1):
            # Convert timestamp to clickable link
            timestamp_with_link = convert_timestamps_to_youtube_links(
                idea.timestamp, video_id
            )

            with st.container(border=True):
                st.subheader(f"{idx}. {idea.title}")
                st.markdown(f"**Description:** {idea.description}")
                st.markdown(f"**Timestamp:** {timestamp_with_link}")


def create_download_buttons(
    subtopics: Subtopics,
    actionable_ideas: ActionableIdeas,
    video_id: str,
    video_url: str = "",
    area_of_life: str = "",
    goal: str = "",
) -> None:
    """Create download buttons for markdown and JSON exports.

    Args:
        subtopics: Extracted subtopics Pydantic model
        actionable_ideas: Extracted actionable ideas Pydantic model
        video_id: YouTube video ID
        video_url: Original YouTube URL (optional)
        area_of_life: User's selected area of life (optional)
        goal: User's specific goal (optional)
    """
    col1, col2 = st.columns(2)

    with col1:
        # Markdown export
        md_content = export_to_markdown(
            subtopics,
            actionable_ideas,
            video_id,
            video_url=video_url,
            area_of_life=area_of_life,
        )
        st.download_button(
            label="📄 Download as Markdown (.md)",
            data=md_content,
            file_name=f"yt-insight-{video_id}.md",
            mime="text/markdown",
        )

    with col2:
        # JSON export
        json_content = export_to_json(
            subtopics,
            actionable_ideas,
            video_url=video_url,
            area_of_life=area_of_life,
            goal=goal,
            pretty=True,
        )
        st.download_button(
            label="📋 Download as JSON (.json)",
            data=json_content,
            file_name=f"yt-insight-{video_id}.json",
            mime="application/json",
        )


def render_history_page() -> None:
    """Render the History page where past analyses are listed and can be viewed.

    Loads entries from the default history store and allows the user to view
    a prior run's results (which populates the UI using existing render
    and export helpers).
    """
    st.title("🕘 Analysis History")
    store = get_history_store()
    try:
        entries = store.load_history()
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return

    if not entries:
        st.info(
            "No history available yet. Run an analysis to create history entries."
        )
        return

    # Show newest first
    for idx, entry in enumerate(reversed(entries), start=1):
        header = f"{idx}. {entry.video_url} — {entry.timestamp}"
        with st.expander(header):
            st.write("**Video URL:**", entry.video_url)
            st.write("**Area of Life:**", entry.area_of_life)
            st.write("**Goal:**", entry.goal or "(none)")
            st.write("**Generated At:**", entry.timestamp)

            cols = st.columns([1, 1, 1])
            if cols[0].button("🔍 View Results", key=f"view_{idx}"):
                # Populate session state and render results
                try:
                    vid = extract_video_id(entry.video_url)
                except Exception:
                    vid = ""

                st.session_state.pipeline_step = "history_loaded"
                st.session_state.final_output = {
                    "video_url": entry.video_url,
                    "area_of_life": entry.area_of_life,
                    "goal": entry.goal,
                    "generated_at": entry.timestamp,
                }

                # Render the stored Pydantic models directly
                render_results_tabs(
                    entry.subtopics, entry.actionable_ideas, vid
                )
                create_download_buttons(
                    entry.subtopics,
                    entry.actionable_ideas,
                    vid,
                    video_url=entry.video_url,
                    area_of_life=entry.area_of_life,
                    goal=entry.goal,
                )

            if cols[1].button("🗑️ Delete", key=f"delete_{idx}"):
                # Deleting a single entry requires rewriting history
                all_entries = list(entries)
                # Compute index in original order
                remove_index = len(entries) - idx
                all_entries.pop(remove_index)
                # Write back
                store.history_file.parent.mkdir(parents=True, exist_ok=True)
                with open(store.history_file, "w", encoding="utf-8") as f:
                    import json as _json

                    _json.dump(
                        [e.model_dump() for e in all_entries], f, indent=2
                    )
                st.success("Entry deleted.")
                st.experimental_rerun()

            if cols[2].button("🧹 Clear All", key=f"clear_{idx}"):
                store.clear_history()
                st.success("History cleared.")
                st.experimental_rerun()


def main() -> None:
    """Main Streamlit application entry point."""
    initialize_session_state()
    # Sidebar navigation
    page = st.sidebar.radio("Page", ["Main App", "History"])

    if page == "History":
        render_history_page()
        return
    st.title("🎬 YT Insight Extractor")
    st.markdown(
        "Provide a YouTube video link, choose an area of life, and optionally add a goal to generate targeted insights."
    )

    with st.form(key="input_form"):
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            value="https://www.youtube.com/watch?v=TrvLEgPpV8s",  # Initial Value
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

        # Run the pipeline for steps 1-3
        display_pipeline_progress(youtube_url, area_of_life, specific_goal)

        if st.session_state.pipeline_step == "step_3_complete":
            st.markdown("---")
            st.subheader("📌 Retrieved RAG Context")
            st.write("**Query:**", st.session_state.final_output["query"])
            for idx, item in enumerate(
                st.session_state.retrieved_context, start=1
            ):
                with st.expander(
                    f"Chunk {idx} — score {item.get('score', 0.0):.3f}"
                ):
                    st.write(item["text"])
                    st.write(
                        "_Chunk start:_",
                        f"{item.get('start_time', 0.0):.1f}s",
                        "_end:_",
                        f"{item.get('end_time', 0.0):.1f}s",
                    )


if __name__ == "__main__":
    main()
