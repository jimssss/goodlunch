from langchain_groq import ChatGroq
import os

chat = ChatGroq(temperature=0.8,
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192"
        )
