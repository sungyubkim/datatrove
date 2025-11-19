from typing import TYPE_CHECKING

from datatrove.pipeline.inference.servers.remote_base import RemoteInferenceServer


if TYPE_CHECKING:
    from datatrove.pipeline.inference.run_inference import InferenceConfig


class VLLMRemoteServer(RemoteInferenceServer):
    """
    Remote vLLM server connector (for direct instantiation only).

    IMPORTANT: This class is NOT used by InferenceRunner's automatic server selection.
    When using InferenceRunner, use server_type="endpoint" to connect to remote vLLM servers.
    This class is provided for advanced users who want to instantiate the server directly.

    The standard way to connect to a remote vLLM server is:
        InferenceRunner(
            config=InferenceConfig(
                server_type="endpoint",  # Use generic endpoint server
                endpoint_url="http://my-vllm-server.com:8000",
                ...
            ),
            ...
        )

    This class exists as a vLLM-specific subclass of RemoteInferenceServer for:
    - Direct instantiation in custom workflows
    - Testing and validation
    - Documentation of vLLM-specific remote server usage

    The generic EndpointServer (used when server_type="endpoint") handles all
    OpenAI-compatible endpoints including vLLM, so this specialized class is
    typically not needed in production pipelines.

    Example (direct instantiation):
        config = InferenceConfig(
            model_name_or_path="meta-llama/Llama-3-8B",
            endpoint_url="http://my-vllm-server.com:8000"
        )
        server = VLLMRemoteServer(config)
    """

    def __init__(self, config: "InferenceConfig"):
        """
        Initialize remote vLLM server connector.

        Args:
            config: InferenceConfig with endpoint_url specified

        Raises:
            ValueError: If endpoint_url is not provided
        """
        if not config.endpoint_url:
            raise ValueError(
                "endpoint_url is required for remote server connection. "
                "Please provide the URL of your remote vLLM server, e.g., "
                "'http://my-server.com:8000'"
            )

        super().__init__(config, endpoint=config.endpoint_url)
