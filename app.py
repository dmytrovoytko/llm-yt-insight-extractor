"""YT Insight Extractor - Main Streamlit Application

Provides a web interface for extracting actionable insights from YouTube videos
using RAG-powered processing pipeline.
"""
import logging
import sys
import os
import streamlit as st
import time
import datetime
import pandas as pd
import plotly.express as px

from core.exports import (
    convert_timestamps_to_youtube_links,
    export_actionable_ideas_to_markdown,
    export_subtopics_to_markdown,
    export_to_json,
    export_to_markdown,
)
from core.generator import generate_all_outputs
from core.prompts import ActionableIdeas, Subtopics
from core.transcript import (
    extract_video_id,
    extract_video_title,
    fetch_youtube_transcript,
)
from core.history import (
    get_history_store,
    load_history,
    save_to_history,
)
from core.llm_config import (
    LLMConfigError,
    create_llm,
)

from core.settings import (
    DEFAULT_OLLAMA_MODEL, DEFAULT_OPENAI_MODEL, DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENROUTER_MODEL,
    # DEFAULT_HF_MODEL,
    DEFAULT_YOUTUBE_URL,
    TOP_K,
)

# TEMP local
DEBUG = True


# logging settings to target the terminal (standard output)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

# a specific logger instance for this script
logger = logging.getLogger(__name__)

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
    "subtopics",
    "actionable_ideas",
    "video_title",
    "video_id",
    "final_output",
    "error_message",
    "llm_instance",
    "use_reranking",
    "feedback_clicked",
]

def initialize_session_state() -> None:
    """Initialize session state variables for pipeline state management."""
    for key in SESSION_STATE_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def run_pipeline_step_1_extract_transcript(
    youtube_url: str,
) -> tuple[list[dict], str, str]:
    """Step 1: Extract and validate transcript.

    Args:
        youtube_url: YouTube video URL.

    Returns:
        Tuple of (transcript_data, video_title, status_message) on success.

    Raises:
        Exception: With error message if extraction fails.
    """
    # Extract video ID
    try:
        video_id = extract_video_id(youtube_url)
    except ValueError as e:
        raise Exception(f"Invalid YouTube URL: {str(e)}")

    # Extract video title
    video_title = extract_video_title(youtube_url)

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
        hours = int(total_duration) // 60 // 60
        minutes = int(total_duration) // 60 - hours * 60
        seconds = int(total_duration) % 60

        status_msg = f"Transcript extracted, {hours:02d}:{minutes:02d}:{seconds:02d}, {total_words} words"
        return transcript_data, video_title, status_msg
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
    mandatory_keyword: str,
    use_reranking: bool,
    rag_engine,
    top_k: int = TOP_K,
) -> tuple[list[dict], Subtopics, ActionableIdeas, str]:
    """Step 3: Retrieve relevant transcript context and generate structured results.

    Args:
        area_of_life: Selected area of life.
        specific_goal: Optional specific goal.
        mandatory_keyword: Optional mandatory keyword.
        rag_engine: Initialized RAG engine.
        top_k: Number of top chunks to retrieve.

    Returns:
        Tuple of (retrieved_context, subtopics, actionable_ideas, status_message).

    Raises:
        Exception: If retrieval or generation fails.
    """
    query = build_rag_query(area_of_life, specific_goal)
    results = rag_engine.retrieve(query, mandatory_keyword, use_reranking=use_reranking, top_k=TOP_K) 

    if not results:
        raise Exception(
            "No relevant context could be retrieved from the transcript."
        )

    subtopics, actionable_ideas = generate_all_outputs(
        area_of_life, specific_goal, results,
        llm_config=st.session_state.llm_instance
    )

    status_msg = (
        f"Retrieved {len(results)} relevant transcript chunks "
        f"and generated structured insights."
    )
    return results, subtopics, actionable_ideas, status_msg

def llm_configuration(llm_instance=None):
    if llm_instance:
        _provider = llm_instance.provider
        _model = llm_instance.model
    else:
        _provider = "Ollama"
        _model = DEFAULT_OLLAMA_MODEL
    return _provider, _model

def display_pipeline_progress(
    youtube_url: str, area_of_life: str, specific_goal: str, mandatory_keyword: str
):
  """Execute and display pipeline progress for steps 1-3.

    Args:
        youtube_url: YouTube video URL.
        area_of_life: Selected area of life.
        specific_goal: Optional specific goal.
    """
  # st.markdown("---")
  # st.subheader("⚙️ Processing Pipeline")
  _provider, _model = llm_configuration(st.session_state.llm_instance)
  st.caption(f"⚙️ Processing Pipeline ({_provider}: {_model})")
  with st.empty(): # to make next elements replace each other = disappear
    # Step 1: Extract Transcript
    start_time = time.perf_counter()
    with st.status(
        "🔄 Step 1: Extracting transcript...", expanded=True
    ) as step1_status:
        try:
            transcript_data, video_title, status_msg = (
                run_pipeline_step_1_extract_transcript(youtube_url)
            )
            st.session_state.transcript_data = transcript_data
            st.session_state.video_title = video_title
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
            # as previous statuses disapper, show extracted video title + duration info
            _title = f"({video_title})" if video_title else ""
            _video_info = status_msg.replace("Transcript extracted", "")
            st.success(
                f"✨ Successfully created {len(chunks)} chunks with embeddings {_title}{_video_info}"
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
            (
                retrieved_context,
                subtopics,
                actionable_ideas,
                status_msg,
            ) = run_pipeline_step_3_rag_and_summarize(
                area_of_life,
                specific_goal,
                mandatory_keyword,
                st.session_state.use_reranking,
                st.session_state.rag_engine,
            )
            st.session_state.retrieved_context = retrieved_context
            st.session_state.subtopics = subtopics
            st.session_state.actionable_ideas = actionable_ideas
            st.session_state.final_output = {
                "query": build_rag_query(area_of_life, specific_goal),
                "retrieved_context": retrieved_context,
                "video_title": st.session_state.video_title,
            }
            st.session_state.pipeline_step = "step_3_complete"

            # log processing duration
            end_time = time.perf_counter()
            processing_time = int(end_time - start_time + 0.5) # floor
            
            try:
                save_to_history(
                    video_url=youtube_url,
                    video_title=st.session_state.video_title or "",
                    area_of_life=area_of_life,
                    goal=specific_goal,
                    subtopics=subtopics,
                    actionable_ideas=actionable_ideas,
                    llm_info=f"{_provider}: {_model}",
                    processing_time=processing_time,
                )
                st.session_state.pipeline_step = "history_saved"
            except Exception as save_error:
                st.warning(
                    f"Results extracted but history could not be saved: {save_error}"
                )

            step3_status.update(
                label=f"✅ Step 3: {status_msg}", state="complete"
            )
            st.success(
                f"🚀 Step 3 complete: {len(retrieved_context)} chunks retrieved and structured results generated."
            )

        except Exception as e:
            st.session_state.error_message = str(e)
            st.session_state.pipeline_step = "error"
            step3_status.update(label=f"❌ Step 3: {str(e)}", state="error")
            st.error(str(e))
            return

    # st.markdown("---")
    st.info(
        f"✅ Pipeline complete in {processing_time} sec. Review retrieved context below." #  and continue to build the LLM summary flow
    )

def save_user_feedback(feedback_value: int =0, entries: list =[], idx: int =0, cur_feedback: int =0):
    if feedback_value != 0:
        logger.info(f"save_user_feedback: {feedback_value=} {idx=} {cur_feedback=}")
        try:
            if not entries:
                # clicked below just generated results - need to read history
                store = get_history_store()
                try:
                    entries = list(store.load_history())
                except Exception as e:
                    st.error(f"Failed to load history: {e}")
                    return

                if not entries:
                    st.info(
                        "No history available yet. Run an analysis to create history entries."
                    )
                    return
                idx = 1 # == latest
            else:
                # clicked from History page in View results
                store = get_history_store() # TEMP FIXME receive as a parameter

            # TODO this part should be in core.history ~ update_last_entry()
            all_entries = list(entries)
            # Compute index in original order
            replace_index = len(entries) - idx
            # remove last entry - we'll re-add it updated
            cur_entry = all_entries.pop(replace_index)
            cur_entry.user_feedback = feedback_value
            all_entries.insert(replace_index, cur_entry) # inserting to the same position
            logger.info(f"\n{cur_entry=}\n{feedback_value=}\n{len(all_entries)=}")
            # Write back
            store.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(store.history_file, "w", encoding="utf-8") as f:
                import json as _json

                _json.dump(
                    [e.model_dump() for e in all_entries], f, indent=2
                )
            
            icon = "👍" if feedback_value==1 else "👎"
            st.success(f"{icon} Your feedback saved!") #  with timestamp: `{cur_entry.timestamp}`
        except Exception as save_error:
            logger.error(f"!!! save_user_feedback: {save_error}")
            st.warning(
                f"History could not be saved: {save_error}"
            )

def render_feedback_buttons(entries: list =[], idx: int =0, cur_feedback: int =0) -> None:
    # ! idx starts with 1
    icon = "👍" if cur_feedback==1 else "👎" if cur_feedback==-1 else "--"
    st.write(f"#### Rate this entry (now {icon})")

    btn_col1, btn_col2, _ = st.columns([1, 1, 2])

    # Thumbs Up Button
    with btn_col1:
        thumbs_up = st.button("👍 Good / Useful", width='stretch', type="primary", 
                                # key=f"feedback-{idx}-up", 
                                on_click=save_user_feedback, args=(+1,), 
                                kwargs={"entries": entries, "idx": idx, "cur_feedback": cur_feedback},
        )

    # Thumbs Down Button
    with btn_col2:
        thumbs_down = st.button("👎 Needs Improvement", width='stretch', 
                                # key=f"feedback-{idx}-down", 
                                on_click=save_user_feedback, args=(-1,), 
                                kwargs={"entries": entries, "idx": idx, "cur_feedback": cur_feedback},
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
    tab1, tab2 = st.tabs(["📚 Subtopics Summary", "💡 Actionable Ideas"])

    with tab1:
        st.markdown("## Subtopics Summary\n")
        for idx, subtopic in enumerate(subtopics.subtopics, start=1):
            # Convert timestamp to clickable link
            timestamp_with_link = convert_timestamps_to_youtube_links(
                subtopic.timestamp, video_id
            )

            with st.container(border=True):
                st.subheader(subtopic.title)
                st.write(subtopic.summary)
                st.markdown(f"**Timestamp:** {timestamp_with_link}")

        # render_feedback_buttons() # if we want separate feedback on subtopics

    with tab2:
        st.markdown("## Actionable Ideas\n")
        for idx, idea in enumerate(actionable_ideas.ideas, start=1):
            # Convert timestamp to clickable link
            timestamp_with_link = convert_timestamps_to_youtube_links(
                idea.timestamp, video_id
            )

            with st.container(border=True):
                st.subheader(f"{idx}. {idea.title}")
                st.markdown(f"{idea.description}")
                st.markdown(f"**Timestamp:** {timestamp_with_link}")
                
        # render_feedback_buttons() # if we want separate feedback on actionable ideas


def create_download_buttons(
    subtopics: Subtopics,
    actionable_ideas: ActionableIdeas,
    video_id: str,
    video_title: str = "",
    video_url: str = "",
    area_of_life: str = "",
    goal: str = "",
    llm_info: str = "",
) -> None:
    """Create download buttons for markdown and JSON exports.

    Args:
        subtopics: Extracted subtopics Pydantic model
        actionable_ideas: Extracted actionable ideas Pydantic model
        video_id: YouTube video ID
        video_url: Original YouTube URL (optional)
        area_of_life: User's selected area of life (optional)
        goal: User's specific goal (optional)
        llm_info: LLM provider, model
    """
    # col1, col2 = st.columns(2)
    col1, col2, col3 = st.columns([1, 1, 2]) # to reduce the gap

    with col1:
        # Markdown export
        md_content = export_to_markdown(
            subtopics,
            actionable_ideas,
            video_id,
            video_title=video_title,
            video_url=video_url,
            area_of_life=area_of_life,
            llm_info=llm_info,
        )
        st.download_button(
            label="📄 Download as Markdown (.md)",
            data=md_content,
            file_name=f"yt-insight-{area_of_life}-{video_id}.md",
            mime="text/markdown",
            on_click="ignore", # Prevents backend script rerun
        )

    with col2:
        # JSON export
        json_content = export_to_json(
            subtopics,
            actionable_ideas,
            video_title=video_title,
            video_url=video_url,
            area_of_life=area_of_life,
            goal=goal,
            llm_info=llm_info,
            pretty=True,
        )
        st.download_button(
            label="📋 Download as JSON (.json)",
            data=json_content,
            file_name=f"yt-insight-{area_of_life}-{video_id}.json",
            mime="application/json",
            on_click="ignore", # Prevents backend script rerun
        )


def render_configuration_page() -> None:
    providers = ["Ollama", "OpenAI", "Anthropic",
            # "HuggingFace",
            "OpenRouter",
    ]
    if st.session_state.llm_instance:
        active_index = providers.index(st.session_state.llm_instance.provider)
    else:
        active_index = len(providers)-1 # OpenRouter
    # 1. Choose LLM Provider
    provider = st.selectbox(
        "Select LLM Provider",
        options=providers,
        index=active_index,
        help="Choose the AI model provider you want to use."
    ).lower()
    
    # 2. API Key Inputs (Conditional)
    # OpenAI Key
    openai_key = None
    if provider == "openai":
        env_openai_key = os.getenv("OPENAI_API_KEY", "")
        openai_key = st.text_input(
            "OpenAI API Key",
            value=env_openai_key,
            type="password",
            help="Leave blank if already set in system environment variables."
        )

    # Anthropic Key
    anthropic_key = None
    if provider == "anthropic":
        env_anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        anthropic_key = st.text_input(
            "Anthropic API Key",
            value=env_anthropic_key,
            type="password",
            help="Leave blank if already set in system environment variables."
        )

    # # HuggingFace Key
    # if provider == "huggingface":
    #     env_hf_key = os.getenv("HUGGINGFACE_API_KEY", "") or os.getenv("HF_API_KEY", "")
    #     hf_key = st.text_input(
    #         "Hugging Face API Token", 
    #         value=env_hf_key, 
    #         type="password",
    #         help="Generate a token with 'Read' permissions in your Hugging Face settings."
    #     )
        
    # OpenRouter Key
    openrouter_key = None
    if provider == "openrouter":
        env_openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        openrouter_key = st.text_input(
            "OpenRouter API Key", 
            value=env_openrouter_key, 
            type="password",
            help="Get a free API key at https://openrouter.ai/keys"
        )


    # 3. Model Selection
    default_models = {
        "ollama": DEFAULT_OLLAMA_MODEL,
        "openai": DEFAULT_OPENAI_MODEL,
        "anthropic": DEFAULT_ANTHROPIC_MODEL,
        # "huggingface": DEFAULT_HF_MODEL,
        "openrouter": DEFAULT_OPENROUTER_MODEL
    }
    model_name = st.text_input(
        "Model Name",
        value=default_models[provider],
        help="Specify the exact model name to use."
    )

    # 4. Connection Button
    connect_btn = st.button("⚡ Connect", width='stretch') # , type="primary"

    if connect_btn:
        try:
            with st.status("Initializing LLM...", expanded=True) as status:
                st.error("")
                st.write(f"Validating {provider.capitalize()} configuration...")
                
                kwargs = {"model": model_name}
                
                # Pass keys if provided in UI (prioritizing UI over env for explicitness)
                if provider == "openai" and openai_key:
                    kwargs["api_key"] = openai_key
                elif provider == "anthropic" and anthropic_key:
                    kwargs["api_key"] = anthropic_key
                # elif provider == "huggingface" and hf_:
                #     kwargs["api_key"] = hf_key
                elif provider == "openrouter" and openrouter_key:
                    kwargs["api_key"] = openrouter_key

                # Create LLM instance using our factory
                st.session_state.llm_instance = create_llm(provider=provider, **kwargs)
                
                st.write("Connection successful!")
                status.update(label="Connection Successful!", state="complete")
                
                # # Clear chat history when switching/connecting LLMs
                # st.session_state.messages = []
                
        except LLMConfigError as e:
            st.session_state.llm_instance = None
            st.error(f"Configuration Error: {str(e)}")
        except Exception as e:
            st.session_state.llm_instance = None
            st.error(f"Unexpected Error: {str(e)}")

    st.divider()
    st.markdown("**Status:**")
    if st.session_state.llm_instance is not None:
        st.success(f"Connected to {provider.capitalize()} ({model_name})")
    else:
        st.warning("Not Connected")


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
        header = f"{idx}. {entry.video_title} — on {entry.area_of_life} — {entry.timestamp:.16} — {entry.llm_info} in {entry.processing_time} sec"
        with st.expander(header):
            st.write("**Video URL:**", entry.video_url)
            if getattr(entry, "video_title", ""):
                st.write("**Title:**", entry.video_title)
            st.write(f"**Area of Life:** {entry.area_of_life}. ", f"**Goal:** {entry.goal}" if entry.goal else "")
            # st.write("**Generated At:**", entry.timestamp) (already in the entry header)

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
                    llm_info=entry.llm_info,
                )
                render_feedback_buttons(entries, idx, entry.user_feedback)

            if cols[1].button("🗑️ Delete", key=f"delete_{idx}"):
                # TODO this part should be in core.history ~ delete_entry()
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
                st.rerun()

            if cols[2].button("🧹 Clear All", key=f"clear_{idx}"):
                store.clear_history()
                st.success("History cleared.")
                st.rerun()


def render_barchart(df, column, subject, orientation, bins=[], labels=[]) -> None:

    if bins:
        # Create a copy to avoid SettingWithCopyWarning on the original dataframe
        plot_df = df.copy()
        
        # Categorize column values into bins
        plot_df[f"{column}_bucket"] = pd.cut(
            plot_df[column], 
            bins=bins, 
            labels=labels, 
            include_lowest=True
        )
        
        # Count records per bin and ensure all labels are present (even if count is 0)
        area_counts = plot_df[f"{column}_bucket"].value_counts().reindex(labels).reset_index()
        area_counts.columns = [subject, 'Number of Records']

        # Create Vertical Bar Chart
        fig = px.bar(
            area_counts,
            x=subject,
            y='Number of Records',
            text='Number of Records',
            title=f"{subject} Distribution",
            color='Number of Records',
            color_continuous_scale='Blues'
        )
    else:
        area_counts = (
            df[column]
            .value_counts()
            .reset_index()
        )
        area_counts = area_counts.sort_values(by="count", ascending=False)
        # rename (column, count) to look understandable on chart
        area_counts.columns = [subject, 'Number of Records']

        fig = px.bar(
            area_counts,
            x='Number of Records', # "count",
            y=subject,
            orientation=orientation,
            text='Number of Records', # "count",
            # TODO instead of renaming try using labels={"count": "Total Records", column: subject}, 
            title=f"{subject} Distribution",
            color=subject, # column,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        height=350,
        xaxis=dict(dtick=1),
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(fig, width='stretch')


def render_piechart(df, column: str, subject: str, label_map: dict ={}, red_green: bool =False):
    """Visualize a column distribution as a pie chart."""

    if not len(df):
        st.info(f"No records available yet.")
        return

    if column not in df.columns or df[column].empty:
        st.info(f"No '{column}' found.")
        return

    # Count occurrences of each value
    area_counts = df[column].value_counts().reset_index()
    area_counts.columns = [column, "count"]

    color_discrete_map = {}
    if label_map:
        # Map technical values to user-friendly labels
        area_counts[f"{column}_label"] = area_counts[column].map(
            lambda x: label_map.get(x, str(x))
        )
        names = f"{column}_label"
        if red_green:
            color_discrete_map={
                list(label_map.keys())[0]: "#e74c3c",
                list(label_map.keys())[1]: "#2ecc71",
            }
    else:
        names = column

    # Create Pie / Donut Chart
    fig = px.pie(
        area_counts,
        values="count",
        names=names, # f"{column}_label", # label_map, # or column
        title=f"{subject} Distribution",
        color=column, # f"{column}_label",
        color_discrete_map=color_discrete_map,
        hole=0.4,  # Makes it a donut chart for better presentation
    )

    if label_map:
        fig.update_traces(textinfo="label+percent", textposition='inside', textfont_size=14,
            hovertemplate="%{label}<br>%{value} entries<br>%{percent}"
            )
    else:
        fig.update_traces(textinfo="value+percent", textposition='inside', textfont_size=14)

    fig.update_layout(
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, width='stretch')


def render_report_dashboard() -> None:
    st.title("📊 Analytics Dashboard")

    store = get_history_store()
    try:
        entries = store.load_history()
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return

    if not entries:
        st.info(
            "No history available yet. Extract some insights to create history entries."
        )
        return
    else:
        df = pd.DataFrame([entry.model_dump() for entry in entries])

        df["area_of_life"] = df["area_of_life"].astype(str)
        df["goal"] = df["goal"].fillna("").astype(str)
        df["llm_info"] = df["llm_info"].fillna("").astype(str)
        df["processing_time"] = pd.to_numeric(
            df["processing_time"], errors="coerce"
        ).fillna(0)

        # ---------------------------------------------------------------------
        # Scorecards Calculation
        # ---------------------------------------------------------------------
        total_records = len(df)
        specified_goals_count = df["goal"].apply(lambda x: len(x.strip()) > 0).sum()
        goal_percentage = (
            (specified_goals_count / total_records * 100) if total_records > 0 else 0
        )

        min_proc_time = df["processing_time"].min()
        max_proc_time = df["processing_time"].max()
        avg_proc_time = df["processing_time"].mean()

        # Scorecards Display
        st.subheader("Key Performance Indicators")
        col_sc1, col_sc2, col_sc3 = st.columns(3)

        with col_sc1:
            st.metric(
                label="Scorecard 1: Total Records",
                value=f"{total_records}",
                delta="Active Items",
            )

        with col_sc2:
            st.metric(
                label="Scorecard 2: Specified Goal Rate",
                value=f"{goal_percentage:.1f}%",
                delta=f"{specified_goals_count} of {total_records} specified",
            )

        with col_sc3:
            st.metric(
                label="Scorecard 3: Processing Time (Avg / Min / Max)",
                value=f"{avg_proc_time:.1f} sec",
                delta=f"Min: {min_proc_time}s | Max: {max_proc_time}s",
                delta_color="off",
            )

        st.markdown("---")

        st.subheader(f"📈 Distribution by Area of Life and ⭕ Goal set / not")
        
        tab1, tab2 = st.columns([3, 1])

        # Chart 1: area_of_life distribution (horizontal bar chart)
        with tab1:
            render_barchart(df, column="area_of_life", subject="Area of Life", orientation="h")

        # Chart 2: area_of_life distribution (pie chart)
        df['goal_set'] = df['goal'] != ""
        with tab2:
            render_piechart(df, column="goal_set", subject="Goal set / not", 
                            label_map={False: "Goal not set", True: "Goal set"}
            )

        st.subheader(f"📈 Distribution by used LLM and Processing Time")

        tab1, tab2 = st.columns([3, 1])
        # Chart 3: llm_info distribution (horizontal bar chart)
        with tab1:
            render_barchart(df, column="llm_info", subject="Used LLM", orientation="h")

        # Chart 4: user_feedback distribution (pie chart)
        with tab2:
            render_piechart(df, column="user_feedback", subject="User feedback", 
                            label_map={0: "No feedback", -1: "Needs Improvement", +1: "Useful"}
            )
        
        # Define bins and human-readable labels
        bins = [0, 30, 60, 90, 120, 360]
        labels = ['0-30s', '31-60s', '61-90s', '91-120s', '121-360s']
        # Chart 5: processing_time distribution (vertical bar chart)
        render_barchart(df, column="processing_time", subject="Processing Time", orientation="v", bins=bins, labels=labels)

        # ---------------------------------------------------------------------
        # Raw Records Inspector
        # ---------------------------------------------------------------------
        # with st.expander("🔍 View Raw Records & Feedback Data"):
        with st.expander("🔍 View Raw Records"):
            st.write("### All Records")
            st.dataframe(
                df[["timestamp", "area_of_life", "goal", "llm_info", "processing_time", "user_feedback"]],
                width='stretch',
            )
                

def main() -> None:
    """Main Streamlit application entry point."""
    initialize_session_state()
    # Sidebar navigation
    page = st.sidebar.radio("Tabs", 
            ["💡 Insights", "🕘 History", "📊 Report Dashboard", "⚙️ Configuration"],
            label_visibility="hidden"
    )

    if page == "🕘 History":
        render_history_page()
        return
    if page == "📊 Report Dashboard":
        render_report_dashboard()
        return
    if page == "⚙️ Configuration":
        render_configuration_page()
        return
    st.title("🎬 YT Insight Extractor")
    st.markdown(
        "Provide a YouTube video link, choose an area of life, and optionally add a goal to generate targeted insights."
    )
    _provider, _model = llm_configuration(st.session_state.llm_instance)
    st.markdown(
        f"Current LLM: {_provider}:{_model}. Use ⚙️ Configuration page to change it if needed."
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
        mandatory_keyword = st.text_input(
            "Your mandatory keyword as a strict content filter (i.e. quick stop, optional)",
            placeholder="E.g. focus, habit, negotiation (at least 4 chars)",
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            use_reranking = st.checkbox("Use re-ranking", value=st.session_state.use_reranking)
            if use_reranking:
                st.session_state.use_reranking = True
            else:
                st.session_state.use_reranking = False

        with col2:
            submit_button = st.form_submit_button("🚀 Process", width="stretch")

        if mandatory_keyword and len(mandatory_keyword)<4:
            st.error("Please enter a longer (4+ chars) mandatory keyword before processing (or empty).")
            return

    if submit_button:
        if not youtube_url:
            st.error("Please enter a YouTube URL before processing.")
            return

        st.session_state.pipeline_step = "inputs_received"
        st.session_state.transcript_data = None
        st.session_state.chunks = None
        st.session_state.retrieved_context = None
        st.session_state.subtopics = None
        st.session_state.actionable_ideas = None
        st.session_state.video_title = None
        st.session_state.video_id = None
        st.session_state.final_output = None
        st.session_state.error_message = None

        try:
            st.session_state.video_id = extract_video_id(youtube_url)
        except ValueError as e:
            st.error(f"Invalid YouTube URL: {str(e)}")
            return

        # st.markdown("---")
        # st.subheader("📝 Received inputs")
        # st.write("**YouTube URL:**", youtube_url)
        # st.write("**Area of Life:**", area_of_life)
        # st.write("**Specific Goal:**", specific_goal or "(none)")

        # Run the pipeline for steps 1-3
        display_pipeline_progress(youtube_url, area_of_life, specific_goal, mandatory_keyword)

        if st.session_state.pipeline_step in ["step_3_complete", "history_saved"]:
            # st.markdown("---")
            st.subheader("✅ Extracted Results")

            if (
                st.session_state.subtopics
                and st.session_state.actionable_ideas
                and st.session_state.video_id
            ):
                st.write("**Video Title:**", st.session_state.video_title)
                render_results_tabs(
                    st.session_state.subtopics,
                    st.session_state.actionable_ideas,
                    st.session_state.video_id,
                )
                create_download_buttons(
                    st.session_state.subtopics,
                    st.session_state.actionable_ideas,
                    st.session_state.video_id,
                    video_title=st.session_state.video_title,
                    video_url=youtube_url,
                    area_of_life=area_of_life,
                    goal=specific_goal,
                )
                render_feedback_buttons()
            else:
                st.warning(
                    "Structured results are not available yet. Please rerun the pipeline or check for errors."
                )

            if DEBUG:
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
