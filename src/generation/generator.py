from collections.abc import Iterator  


#Sends the prompt to the LLM en awaits the complete answer
def generate(llm, messages) -> str:
    return llm.invoke(messages).content  


#Returns the answer piece by piece once the LLM starts writing it
def generate_stream(llm, messages) -> Iterator[str]:
    for chunk in llm.stream(messages):  
        yield chunk.content  