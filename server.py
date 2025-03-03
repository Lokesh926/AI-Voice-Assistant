from pathlib import Path

from fastapi import Depends, FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from groq import AsyncGroq
from pydantic_ai import Agent 

from src.app.lifespan import app_lifespan as lifespan
from src.app.llm import Dependencies
from src.app.settings import get_settings
from src.app.stt import transcribe_audio_data
from src.app.tts import TextToSpeech


app = FastAPI(title='Voice to Voice Demo', lifespan=lifespan) #1. Lifespan

@app.get("/")

async def get():
    with Path("simple_ui.html").open("r") as file:
        return HTMLResponse(file.read())  # This will render the HTML


#2. FastAPI Dependency Injection

async def get_agent_dependencies(websocket: WebSocket) -> Dependencies:
    return Dependencies (
        settings= get_settings(),
        session= websocket.state.aiohttp_session,
    )

async def get_groq_client(websocket: WebSocket) -> AsyncGroq:
    return websocket.state.groq_client

async def get_agent(websocket: WebSocket) -> Agent:
    return websocket.state.groq_agent

async def get_tts_handler(websocket:WebSocket) -> TextToSpeech:
    return TextToSpeech(
        client=websocket.state.openai_client,
        model_name='tts-1',
        response_format='aac',
    )

#3. WebSocket

@app.websocket('/voice_stream')

async def voice_to_voice(
    websocket: WebSocket,
    groq_client: AsyncGroq= Depends(get_groq_client),
    agent: Agent[Dependencies] = Depends(get_agent),
    agent_deps: Dependencies = Depends(get_agent_dependencies),
    tts_handler: TextToSpeech= Depends(get_tts_handler),

):
    
    """
    WebSocket endpoint for voice-to-voice communication.

    - Recieves audio bytes from the client.
    - Transcribe the audio to text 
    - generated a response using the language model agent
    - converts the response text to speech, and streams the audio bytes back to the client.
    
    Args:
        websocket: WebSocket connection.
        coversation_id : Unique identifier for the conversation (dependency).
        db_conn: Asynchronous database connection (dependency).
        groq_client: Groq API client for transcription (dependency).
        agent: Language model agent for generating responses (dependency).
        agent_deps: Dependencies for the agent (dependency).
        tts_handler: Text-to-Speech handler for converting text to audio (dependency).
    
    """

    await websocket.accept()

    async for incoming_audio_bytes in websocket.iter_bytes():
        #step 1: Transcribe the incoming audio

        transcription = await transcribe_audio_data(
            audio_data=incoming_audio_bytes,
            api_client=groq_client
        )

        #step 2: Generate the agent's response

        generation = ""
        async with tts_handler:
            async with agent.run_stream(
                user_prompt=transcription,
                deps= agent_deps
            ) as results:
                async for message in results.stream_text(delta=True):
                    generation += message

                # Stream the audio back to the client
                async for audio_chunk in tts_handler.feed(text=message):
                    await websocket.send_bytes(data=audio_chunk)

            # Flush any remaining audio chunks 

            async for audio_chunks in tts_handler.flush():
                await websocket.send_bytes(data=audio_chunk)