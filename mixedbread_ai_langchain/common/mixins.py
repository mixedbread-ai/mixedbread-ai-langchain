"""Mixins for common component functionality."""

from typing import Dict, Any, Type
from pydantic import SecretStr

from .client import MixedbreadClient


class SerializationMixin:
    """Mixin to provide standard serialization/deserialization for components."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize component to dictionary.

        Returns:
            Dictionary representation of the component.
        """
        if not isinstance(self, MixedbreadClient):
            raise TypeError(
                "SerializationMixin can only be used with MixedbreadClient subclasses"
            )

        client_params = MixedbreadClient.to_dict(self)

        # Get all component-specific parameters
        component_params = {}
        for attr in dir(self):
            if not attr.startswith("_") and hasattr(self, attr):
                value = getattr(self, attr)
                # Skip methods and client-specific attributes
                if not callable(value) and attr not in [
                    "client",
                    "async_client",
                    "api_key",
                    "base_url",
                    "timeout",
                    "max_retries",
                ]:
                    component_params[attr] = value

        return {
            "type": self.__class__.__name__,
            "init_parameters": {
                **client_params,
                **component_params,
            },
        }

    @classmethod
    def from_dict(cls: Type, data: Dict[str, Any]):
        """
        Deserialize component from dictionary.

        Args:
            data: Dictionary containing component data.

        Returns:
            Instantiated component.
        """
        init_params = data.get("init_parameters", {})

        # Handle api_key deserialization
        api_key = init_params.get("api_key")
        if isinstance(api_key, dict):
            init_params["api_key"] = SecretStr(api_key)

        return cls(**init_params)


class AsyncMixin:
    """Mixin to provide async support patterns."""

    def _ensure_async_client(self):
        """Ensure async client is available."""
        if not hasattr(self, "_client") or not hasattr(self._client, "async_client"):
            raise RuntimeError(
                "Async client not available. Component must inherit from MixedbreadClient."
            )

        return self._client.async_client


class ErrorHandlingMixin:
    """Mixin to provide consistent error handling patterns."""

    def _handle_api_error(
        self, error: Exception, operation: str, fallback_result: Any = None
    ):
        """
        Handle API errors consistently across components.

        Args:
            error: The exception that occurred.
            operation: Description of the operation that failed.
            fallback_result: Fallback result to return on error.

        Returns:
            Fallback result or re-raises the error.
        """
        from .logging import get_logger

        logger = get_logger()
        logger.error(f"Error during {operation}: {str(error)}")

        if fallback_result is not None:
            logger.warning(f"Returning fallback result for {operation}")
            return fallback_result

        raise error
