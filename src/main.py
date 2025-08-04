from tutorial.basic_agent import graph
import os
from phoenix.otel import register

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")  

# Setup LLM Observability
tracer_provider = register(
  project_name="my-llm-app", # Default is 'default'
  auto_instrument=True, # See 'Trace all calls made to a library' below
)

tracer = tracer_provider.get_tracer(__name__)

@tracer.chain
def stream_graph_updates(user_input: str):
    """Send user input to the agent and stream assistant responses to stdout."""
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            # Each event may contain partial or complete assistant messages.
            print("Assistant:", value["messages"][-1].content)


if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except (KeyboardInterrupt, EOFError):
            # Gracefully handle interruption or non-interactive environment
            print("\nGoodbye!")
            break
        except Exception as exc:
            # Fallback example question if stdin is not available or other errors occur
            fallback_query = "What do you know about LangGraph?"
            print(f"Encountered error: {exc}. Using fallback question.\nUser: {fallback_query}")
            stream_graph_updates(fallback_query)
            break
