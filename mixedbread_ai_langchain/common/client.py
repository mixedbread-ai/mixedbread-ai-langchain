import os
from typing import Optional
import httpx
from mixedbread import Mixedbread
from mixedbread import AsyncMixedbread

USER_AGENT_VERSION = "0.1.0"
USER_AGENT = f"mixedbread-ai-langchain/{USER_AGENT_VERSION}"


class MixedbreadClient:
    """
    Shared client for Mixedbread AI API services.
    Handles API key management and SDK client initialization.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
        _httpx_client: Optional[httpx.Client] = None,
        _async_httpx_client: Optional[httpx.AsyncClient] = None,
    ):

        self.api_key = api_key or os.getenv("MXBAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mixedbread API key not found. Please set the MXBAI_API_KEY environment variable or pass it directly."
            )

        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        self._client = Mixedbread(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client=_httpx_client,
            default_headers={"User-Agent": USER_AGENT},
        )

        self._async_client = AsyncMixedbread(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client=_async_httpx_client,
            default_headers={"User-Agent": USER_AGENT},
        )

    @property
    def client(self) -> Mixedbread:
        return self._client

    @property
    def async_client(self) -> AsyncMixedbread:
        return self._async_client
