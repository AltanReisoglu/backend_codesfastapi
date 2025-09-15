import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = "AIzaSyBymPtSh5RXrPVn4UfXcDlHV_M7nW8X3yA"

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage



from google import genai
#from google.genai import types
import base64

"""def generate():

  client = genai.Client(
      vertexai=True,
      project="246624734452",
      location="us-central1",
  )


  model = "projects/246624734452/locations/us-central1/endpoints/3176166935737925632"
  contents = [
    types.Content(
      role="user",
      parts=[
        #types.Part.from_text(text=merhaba)
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk.text, end="")

generate()"""

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

