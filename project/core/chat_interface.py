from langchain_core.messages import HumanMessage


class ChatInterface:

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def chat(self, message, history):
        if not self.rag_system.agent_graph:
            return "⚠️ System not initialized!"

        query = message.strip()
        cache = getattr(self.rag_system, "cache", None)

        # Check Redis LLM cache before invoking the graph
        if cache:
            cached = cache.get_llm_response(query)
            if cached:
                return cached

        try:
            result = self.rag_system.agent_graph.invoke(
                {"messages": [HumanMessage(content=query)]},
                self.rag_system.get_config(),
            )
            response = result["messages"][-1].content

            if cache:
                cache.set_llm_response(query, response)

            return response

        except Exception as e:
            return f"❌ Error: {str(e)}"
        finally:
            self.rag_system.observability.flush()

    def clear_session(self):
        self.rag_system.reset_thread()
