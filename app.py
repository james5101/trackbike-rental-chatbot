import streamlit as st
import os
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
import pinecone

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment='us-east-1-aws',
)
index_name = "trackbike-rentals"
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

  

loader = DirectoryLoader('./docs', glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings()
with st.spinner('Loading embeddings...'):
    index = Pinecone.from_existing_index(index_name, embeddings)
# index = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    st.success("Embeddings done.", icon="✅")


qa = RetrievalQA.from_chain_type(
                llm=OpenAI(openai_api_key=os.environ['OPENAI_API_KEY']),
                chain_type = "map_reduce",
                retriever=index.as_retriever(),
)

tools = [
    Tool(
        name="Trackbike Rental QA System",
        func=qa.run,
        description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
    )
]
prefix = """Have a conversation with a human, answering the following questions as best you can based only on the context and memory available. 
            You have access to a single tool:"""
suffix = """Go!"
{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history"
    )

llm_chain = LLMChain(
    llm=OpenAI(
        temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'], model_name="text-davinchi-003"
    ),
    prompt=prompt,
)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
)

st.title("🏍️TrackBike Rental's Chatbot")
query = st.text_input("What's your question?",placeholder="Ask me anything about suspension",max_chars=1000)


            

if query:
    with st.spinner(
        "Generating Answer to your Question : `{}` ".format(query)
    ):
        res = agent_chain.run(query)
        st.info(res, icon="🏍️")

with st.expander("History/Memory"):
                st.session_state.memory


