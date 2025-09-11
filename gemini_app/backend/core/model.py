from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

#from core.prompts import STORY_PROMPT
#from core.models import StoryLLMResponse, StoryNodeLLM


from dotenv import load_dotenv
import os

load_dotenv()


class StoryGenerator:

    @classmethod
    def _get_llm(cls):
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

    @classmethod
    def generate_story(cls, theme: str = "fantasy") -> None:
        llm = cls._get_llm()
        #story_parser = PydanticOutputParser(pydantic_object=StoryLLMResponse)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "you are bot that creates choose-your-own-adventure stories"
            ),
            (
                "human",
                f"Create the story with this theme: {theme}"
            )
        ])#.partial(format_instructions=story_parser.get_format_instructions())

        # LLM'den raw response al
        raw_response = llm.invoke(prompt.invoke({}))

        response_text = raw_response
        if hasattr(raw_response, "content"):
            response_text = raw_response.content

        # Pydantic parse et
        #story_structure = story_parser.parse(response_text)

        return response_text
#print(StoryGenerator.generate_story())