import os
import requests
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

load_dotenv()
nest_asyncio.apply()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

my_agent = Agent(
    model="groq:llama-3.1-8b-instant",
    model_settings=ModelSettings(temperature=0.2),
    output_type=str,
    system_prompt="You are a weather assistant. Use get_weather tool to answer weather questions."
)

@my_agent.tool
def get_weather(ctx: RunContext, city: str) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': os.getenv("WEATHER_API_KEY"),
        'units': 'metric'
    }
    raw_data = requests.get(url, params=params).json()
    return {
        "city": raw_data["name"],
        "sky": raw_data["weather"][0]["description"].capitalize(),
        "temp": raw_data["main"]["temp"]
    }

st.title("🌦️ AI Weather Assistant")
st.write("Ask me anything about weather!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask about weather...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Fetching weather..."):
            try:
                result = my_agent.run_sync(question)
                st.write(result.output)
                st.session_state.messages.append({"role": "assistant", "content": result.output})
            except Exception as e:
                st.error(f"Error: {str(e)}")