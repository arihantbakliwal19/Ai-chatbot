from langchain_groq import ChatGroq

def get_chatgroq_model():

    api_key = "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"

    model = ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant"
    )

    return model