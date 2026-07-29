import streamlit as st

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
    "retrieved_context",
    "final_output",
    "error_message",
]


def initialize_session_state() -> None:
    for key in SESSION_STATE_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def main() -> None:
    st.set_page_config(page_title="YT Insight Extractor", layout="wide")
    initialize_session_state()
    st.title("YT Insight Extractor")
    st.markdown(
        "Provide a YouTube video link, choose an area of life, and optionally add a goal to generate targeted insights."
    )

    with st.form(key="input_form"):
        youtube_url = st.text_input(
            "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
        )
        area_of_life = st.selectbox("Area of Life", AVAILABLE_AREAS)
        specific_goal = st.text_input(
            "Optional specific goal",
            placeholder="E.g. improve focus, build better habits, learn negotiation",
        )
        submit_button = st.form_submit_button("Process")

    if submit_button:
        if not youtube_url:
            st.error("Please enter a YouTube URL before processing.")
            return

        st.session_state.pipeline_step = "inputs_received"
        st.session_state.transcript_data = None
        st.session_state.retrieved_context = None
        st.session_state.final_output = None
        st.session_state.error_message = None

        st.success("Inputs received. Running the MVP workflow...")
        st.markdown("---")
        st.subheader("Received inputs")
        st.write("**YouTube URL:**", youtube_url)
        st.write("**Area of Life:**", area_of_life)
        st.write("**Specific Goal:**", specific_goal or "(none)")


if __name__ == "__main__":
    main()
