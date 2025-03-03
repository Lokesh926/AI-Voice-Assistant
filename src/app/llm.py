from dataclasses import dataclass
from typing import Literal

import aiohttp
from groq import AsyncGroq
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.groq   import GroqModel

from settings import Settings

type AvailableCities = Literal['Paris','Madrid','London']


#1. Dependencies:
@dataclass
class Dependencies: 
    settings: Settings
    session: aiohttp.ClientSession

#2. Tool: We can define functions that can be used by the agent to perform specific tasks.

async def get_weather(ctx: RunContext[Dependencies],city: AvailableCities) -> str:
    url = 'http://api.weatherstack.com/current'
    params = {
        'access_key': ctx.deps.settings.weatherstack_api_key,
        'query':  city,
    }
    async with ctx.deps.session.get(url=url, params=params) as response:
        data = await response.json()
        observation_time = data.get("current").get("observation_time")
        temperature = data.get("current").get("temperature")
        weather_descriptions = data.get("current").get("weather_descriptions")
        return f"At {observation_time},the temperature in {city} is {temperature} degrees celcius. The weather is {weather_descriptions[0].lower()}"
    
#3. Model 

def create_groq_model(
        groq_client: AsyncGroq 
        ) -> GroqModel:
    return GroqModel(model_name='llama-3.3-70b-versatile')


#4 Agent
def create_groq_agent(
        groq_model: GroqModel,
        tools: list[Tool[Dependencies]],
        system_prompt: str,  #5. System Prompt
) -> Agent[Dependencies]:
    return Agent(
        model=groq_model,
        deps_type = Dependencies,
        system_prompt = system_prompt,
        tools= tools,
    )
