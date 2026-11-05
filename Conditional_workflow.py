
import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
import streamlit as st
import uuid
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# --------------------------------
# 1. LLM
# --------------------------------

llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
    max_tokens=1024
)


# --------------------------------
# 2. Embeddings
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# 3. PDF Retriever Function
# --------------------------------

def build_retriever(pdf_path: str):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever


# Create separate retrievers

academic_retriever = build_retriever("academics_handbook.pdf")

fee_retriever = build_retriever("fee_structure.pdf")


# --------------------------------
# 4. Define Graph State
# --------------------------------

class CollegeState(TypedDict):

    programme: str

    messages: Annotated[list[AnyMessage], add_messages]

    query_type: str

    retrieved_context: str


# --------------------------------
# 5. Classifier Node
# --------------------------------

def classifier_node(state: CollegeState) -> dict:

    last_message = state["messages"][-1].content

    prompt = f"""
    Classify the following student query into exactly one category:

    academic, fee, or general.

    Academic:
    Attendance, exams, grading, credits, promotion,
    course structure, degree requirements.

    Fee:
    Tuition, payment, refund, late charges,
    scholarships, or money-related questions.

    General:
    Greetings, casual talk, or unrelated questions.

    Query: {last_message}

    Return only one word:
    academic, fee, or general.
    """

    response = llm.invoke(prompt)

    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"

    elif "fee" in category:
        category = "fee"

    else:
        category = "general"

    return {"query_type": category}


# --------------------------------
# 6. Academic RAG Node
# --------------------------------

def academic_rag_node(state: CollegeState) -> dict:

    query = state["messages"][-1].content

    docs = academic_retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {"retrieved_context": context}


# --------------------------------
# 7. Fee RAG Node
# --------------------------------

def fee_rag_node(state: CollegeState) -> dict:

    query = state["messages"][-1].content

    docs = fee_retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {"retrieved_context": context}


# --------------------------------
# 8. General Node
# --------------------------------

def general_node(state: CollegeState) -> dict:

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


# --------------------------------
# 9. Response Node
# --------------------------------

def response_node(state: CollegeState) -> dict:

    query = state["messages"][-1].content

    programme = state.get("programme", "Unknown")

    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = f"""
        You are a friendly college assistant.

        You are talking to a {programme} student.

        Answer using general knowledge.

        Question:
        {query}
        """

    else:

        prompt = f"""
        You are a college assistant helping a {programme} student.

        Answer ONLY using the following official college documents.

        If the answer is not available in the context,
        clearly say that the information is not available.

        Highlight information relevant to {programme}.

        Context:
        {context}

        Question:
        {query}

        Give a clear and precise answer.
        """

    response = llm.invoke(prompt)

    return {
        "messages": [
            ("ai", response.content.strip())
        ]
    }


# --------------------------------
# 10. Conditional Routing
# --------------------------------

def route_query(state: CollegeState):

    if state["query_type"] == "academic":

        return "academic"

    elif state["query_type"] == "fee":

        return "fee"

    else:

        return "general"


# --------------------------------
# 11. Build Graph
# --------------------------------

graph = StateGraph(CollegeState)


# Register nodes

graph.add_node("classifier", classifier_node)

graph.add_node("academic", academic_rag_node)

graph.add_node("fee", fee_rag_node)

graph.add_node("general", general_node)

graph.add_node("response", response_node)


# Define edges

graph.add_edge(START, "classifier")


graph.add_conditional_edges(
    "classifier",
    route_query,
    {
        "academic": "academic",
        "fee": "fee",
        "general": "general"
    }
)


graph.add_edge("academic", "response")

graph.add_edge("fee", "response")

graph.add_edge("general", "response")

graph.add_edge("response", END)


# Compile graph

memory = MemorySaver()

app =graph.compile(checkpointer=memory)


# --------------------------------
# 12. User Interface
# --------------------------------


# --------------------------------
# 12. STREAMLIT UI
# --------------------------------

st.set_page_config(
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="centered"
)

# Initialize session state

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# Custom CSS

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


# Sidebar

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

        # Create a new conversation thread
        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()

    st.divider()

    st.caption("Powered by LangGraph + LangChain + OpenRouter")


# Main UI

st.markdown(
    '<div class="main-title">🎓 College AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask anything about academics, fees and college rules.</div>',
    unsafe_allow_html=True
)


# Welcome message

if not st.session_state.chat_history:

    st.info(
        f"Welcome! You are currently registered as a "
        f"**{student_programme}** student. How can I help you today?"
    )


# Display previous chat messages

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# Chat input

user_query = st.chat_input(
    "Ask your question about college..."
)


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
