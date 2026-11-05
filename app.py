
import streamlit as st
import uuid

from Conditional_workflow import app


# --------------------------------
# 1. Page Config
# --------------------------------
st.set_page_config(
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="centered"
)


# --------------------------------
# 2. Session State
# --------------------------------
if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(uuid.uuid4())


# --------------------------------
# 3. Custom CSS
# --------------------------------
st.markdown("""
<style>

    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 25px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------
# 4. Sidebar
# --------------------------------
with st.sidebar:

    st.title("🎓 College Assistant")

    st.write("Your AI-powered college help desk.")

    st.divider()

    st.subheader("Student Information")

    student_programme = st.selectbox(
        "Select your programme",
        ["BCA", "BBA", "B.Com (H)"]
    )

    st.divider()

    st.subheader("Chat Settings")

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.chat_history = []

        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()

    st.divider()

    st.caption("Powered by LangGraph + LangChain + OpenRouter")


# --------------------------------
# 5. Main UI
# --------------------------------
st.markdown(
    '<div class="main-title">🎓 College AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask anything about academics, fees and college rules.</div>',
    unsafe_allow_html=True
)


# --------------------------------
# 6. Welcome Message
# --------------------------------
if not st.session_state.chat_history:

    st.info(
        f"Welcome! You are currently registered as a "
        f"**{student_programme}** student. How can I help you today?"
    )


# --------------------------------
# 7. Display Previous Chat Messages
# --------------------------------
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------
# 8. Chat Input
# --------------------------------
user_query = st.chat_input(
    "Ask your question about college..."
)


# --------------------------------
# 9. Run LangGraph
# --------------------------------
if user_query:

    # Display user message

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):

        st.markdown(user_query)


    # Get response from LangGraph

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = app.invoke(

                    {
                        "programme": student_programme,

                        "messages": [
                            ("human", user_query)
                        ]
                    },

                    config={

                        "configurable": {

                            "thread_id": st.session_state.thread_id

                        }

                    }

                )

                answer = result["messages"][-1].content

                st.markdown(answer)

                st.session_state.chat_history.append({

                    "role": "assistant",

                    "content": answer

                })

            except Exception as e:

                st.error(f"Something went wrong: {str(e)}")
