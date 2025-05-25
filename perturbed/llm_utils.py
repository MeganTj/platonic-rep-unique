import os
from typing import TypeVar, TypedDict
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
ResponseFormatT = TypeVar("ResponseFormatT")

class FormatException(Exception):
    """Exception raised for errors in the format."""
    def __init__(self, message="Format is incorrect"):
        self.message = message
        super().__init__(self.message)

class Captions(BaseModel):
    unique_captions: list[str]
    
    class Config:
        extra = "forbid"

class Message(TypedDict):
  role: str
  content: str

def create_message(role: str, content: str) -> Message:
  return {
    "role": role,
    "content": content
  }

class LLM:
  def __init__(self, llm_str: str = "gpt-4o-mini", default_instructions: str | None = None, 
               timeout: int | None = None, use_async=False):
    client_class = AsyncOpenAI if use_async else OpenAI
    self.client = client_class(
      api_key=os.getenv("MIT_OPENAI_API_KEY"),
      timeout=timeout
    )
    self.llm_str = llm_str
    self.instructions = default_instructions

  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1)) 
  async def parse_completion_async(self, messages, response_format: ResponseFormatT, **kwargs):
    instructions = kwargs.pop("instructions", self.instructions)

    if instructions is not None:
      system_message = create_message("system", instructions)
      messages = [system_message, *messages]
    response = await self.client.beta.chat.completions.parse(
        model=self.llm_str,
        messages=messages,
        response_format=response_format,
        **kwargs,
        )
    return response

  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)) 
  def parse_completion(self, messages, response_format: ResponseFormatT, **kwargs):
    instructions = kwargs.pop("instructions", self.instructions)

    if instructions is not None:
        system_message = create_message("system", instructions)
        messages = [system_message, *messages]
    response = self.client.beta.chat.completions.parse(
        model=self.llm_str,
        messages=messages,
        response_format=response_format,
        **kwargs,
        )
    return response
  
def generate_captions(prompt, default_instructions=None, **kwargs):
    llm = LLM(llm_str = "gpt-4o-mini", default_instructions=default_instructions) #gpt4o
    response = llm.parse_completion(prompt, response_format=Captions, **kwargs)
    return response

async def generate_captions_async(prompt, default_instructions=None, **kwargs):
    llm = LLM(llm_str = "gpt-4o-mini", default_instructions=default_instructions, use_async=True) #gpt4o
    response = await llm.parse_completion_async(prompt, response_format=Captions, **kwargs)
    return response