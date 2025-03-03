from typing import AsyncIterator, Literal
from openai import AsyncOpenAI
from types import TracebackType


type Voice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
type ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

class TextToSpeech:
    def __init__(self,
                 client: AsyncOpenAI,
                 model_name: str,
                 voice: Voice ='echo',
                 response_format: ResponseFormat = "aac",
                 speed : float = 1.00,
                 buffer_size: int = 64,
                 sentence_endings: tuple[str, ...] =
                 (
                     '.',
                     '?',
                     '!',
                     ';',
                     ':',
                     '\n',

                 ),
                 chunk_size : int = 1024*5,

                 ) -> None:
        self.client = client
        self.model_name = model_name
        self.voice: Voice = voice
        self.response_format: ResponseFormat = response_format
        self.speed = speed
        self.buffer_size = buffer_size
        self.sentence_endings = sentence_endings
        self.chunk_size = chunk_size
        self._buffer = ""


    async def __aenter__(self) -> "TextToSpeech":
        """
        Enter the asynchronous context manager

        Returns:
            The TextToSpeech instance.

        """
        return self

    
    async def feed(self, text:str)-> AsyncIterator[bytes]:
        
        """
        Feed text into buffer and yields audio bytes if the buffer reaches the buffer size or ends with a sentence-ending character.

        Args: 
            text: The text to add to the buffer.

        Yields:
            Audio bytes generated from the buffered text.
        
        """

        self._buffer += text
        if len(self._buffer) >= self.buffer_size or any(
            self._buffer.endswith(se) for se in self.sentence_endings
        ):
            async for chunk in self.flush():
                yield chunk

        
    async def flush(self) -> AsyncIterator[bytes]:
        """
        Flushes the buffered text and yields the resulting audio bytes

        Yields:
            Audio bytes generated from the bufferd text.

        """

        if self._buffer:
            async for chunk in self._send_audio(self._buffer):
                yield chunk

            self._buffer = ""

    
    async def _send_audio(self, text: str) -> AsyncIterator[bytes]:
        """
        Send text to the TTS API and yields audio chunks.

        Args:
            text: The text to convert to speech 

        Yield:
            Chunk of audio bytes generated from the text.
        
        """

        async with self.client.audio.speech.with_streaming_response.create(

        model= self.model_name,
        input= text,
        voice=self.voice,
        response_format= self.response_format,
        speed= self.speed,
        ) as audio_stream:
            async for audio_chunk in audio_stream.iter_bytes(
                chunk_size= self.chunk_size
            ):
                yield audio_chunk

    async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            exc_tb : TracebackType | None,
    ) -> None:
        
        """
        Exits the asynchronous context manager
        
        """

        pass


    

        