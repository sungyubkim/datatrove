"""
Inference pipeline for running LLM inference on documents.

This module provides infrastructure for running inference on documents using various
inference servers like SGLang and VLLM. It supports concurrent processing, metrics
collection, and post-processing steps.

Parts of this implementation are adapted from https://github.com/allenai/olmocr
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Iterable, Literal

from loguru import logger

from datatrove.data import Document
from datatrove.io import get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.metrics import MetricsKeeper, QueueSizesKeeper
from datatrove.pipeline.inference.servers import (
    DummyServer,
    InferenceServer,
    SGLangServer,
    VLLMRemoteServer,
    VLLMServer,
)
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.writers.disk_base import DiskWriter


def reconstruct_gptoss_from_vllm_response(choice: dict) -> str:
    """
    Reconstruct original GPT OSS format from vLLM's parsed response.

    When vLLM's reasoning parser is enabled, it automatically parses GPT OSS format into:
    - message.reasoning_content: Analysis channel content (<|channel|>analysis)
    - message.content: Final channel content (<|channel|>final)
    - message.tool_calls: Tool invocations (to=functions.X)

    This function reconstructs the original GPT OSS format that scorers expect,
    which includes all the special tokens (<|start|>, <|channel|>, <|message|>, etc.).

    Args:
        choice: Response choice dict from vLLM (response["choices"][0])

    Returns:
        Reconstructed GPT OSS format string, or original content if not parsed

    Example:
        Input (vLLM parsed):
            {
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42",
                    "reasoning_content": "Let me think... 6*7=42"
                }
            }

        Output (reconstructed):
            <|start|>assistant<|channel|>analysis<|message|>Let me think... 6*7=42<|end|>
            <|start|>assistant<|channel|>final<|message|>The answer is 42<|return|>
    """
    message = choice.get("message", {})
    reasoning_content = message.get("reasoning_content")
    content = message.get("content")
    tool_calls = message.get("tool_calls")

    # If no reasoning_content or tool_calls, vLLM parser wasn't active
    # Return original content as-is
    if reasoning_content is None and tool_calls is None:
        return content or ""

    # Reconstruct GPT OSS format from parsed components
    parts = []

    # 1. Analysis channel (thinking/reasoning)
    if reasoning_content:
        parts.append(
            f"<|start|>assistant<|channel|>analysis<|message|>"
            f"{reasoning_content}<|end|>"
        )

    # 2. Tool calls
    if tool_calls:
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = function.get("name", "unknown")
            tool_args = function.get("arguments", "{}")
            parts.append(
                f"<|start|>assistant to=functions.{tool_name}"
                f"<|channel|>commentary json<|message|>"
                f"{tool_args}<|call|>"
            )

    # 3. Final channel (only if no tool calls)
    # Tool calls typically don't include final channel
    if content and not tool_calls:
        parts.append(
            f"<|start|>assistant<|channel|>final<|message|>"
            f"{content}<|return|>"
        )

    return "\n".join(parts)


@dataclass
class InferenceSuccess:
    """
    Successful inference result.

    Attributes:
        text: Generated text from the model
        finish_reason: Reason why generation finished
        usage: Token usage statistics from the model
    """

    text: str
    finish_reason: str
    usage: dict


@dataclass
class InferenceError:
    """
    Failed inference result.

    Attributes:
        error: Error message describing what went wrong
    """

    error: str


class InferenceProcessingError(Exception):
    """
    Exception raised when document inference processing fails.

    Attributes:
        document: The original document that failed processing
        error: The underlying error that caused the failure
    """

    def __init__(self, document: Document, error: str | Exception):
        self.document = document
        self.error = error
        super().__init__(f"Failed to process document {document.id}: {error}")


# --------------------------------------------------------------------------- #
# Low-level, dependency-free HTTP POST helper (kept from the original file)
# --------------------------------------------------------------------------- #
async def _raw_post(url: str, json_data: dict) -> tuple[int, bytes]:
    """
    Very small HTTP/1.1 POST helper using the std-lib socket machinery.

    Args:
        url: The target URL for the POST request
        json_data: Dictionary to be sent as JSON payload

    Returns:
        Tuple of (status_code, response_body)
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port or 80
    path = parsed.path or "/"

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    reader, writer = await asyncio.open_connection(host, port)
    try:
        payload = json.dumps(json_data).encode()
        request = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Connection: close\r\n\r\n"
        ).encode()
        writer.write(request + payload)
        await writer.drain()

        # Status line
        status_parts = (await reader.readline()).decode().split(" ", 2)
        status_code = int(status_parts[1]) if len(status_parts) >= 2 else 500

        # Headers (ignored – we rely on Content-Length only)
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break

        # Body
        body = await reader.read()  # connection closes -> EOF
        return status_code, body
    finally:
        writer.close()
        await writer.wait_closed()


# --------------------------------------------------------------------------- #
# Public, simplified configuration
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    """
    Configuration for inference server and processing parameters.

    Attributes:
        server_type: Type of inference server to use
            - "vllm": Local vLLM server
            - "sglang": Local SGLang server
            - "dummy": Local dummy server for testing
            - "vllm-remote": External vLLM server (requires external_endpoint)
        model_name_or_path: Path or name of the model to load
        temperature: Sampling temperature for generation
        model_max_context: Maximum context length for the model
        max_concurrent_requests: Maximum number of concurrent requests to server
        max_concurrent_tasks: Maximum number of concurrent processing tasks
            If your query_builder is slow, it's better to provide higher value than concurrent requests
            to ensure that there are always enough requests to keep the server busy.
            If not provided, will be set to max_concurrent_requests.
        metric_interval: Interval for metrics reporting in seconds
        tp: Tensor parallelism size (number of GPUs to use). Automatically converted to
            --tensor-parallel-size for VLLM or --tp-size for SGLang. Default is 1 (no parallelism)
        dp: Data parallelism size (number of full model replicas). Each replica can span multiple GPUs
            if tensor parallelism is also used. Automatically converted to --data-parallel-size for VLLM
            or --dp-size for SGLang. Default is 1 (no parallelism)
        pp: Pipeline parallelism size (number of pipeline stages). Model layers are distributed across
            pipeline stages for processing in sequence. Automatically converted to --pipeline-parallel-size
            for VLLM or --pp-size for SGLang. Default is 1 (no parallelism)
        use_chat: Whether to use chat format (/v1/chat/completions) or completion format (/v1/completions).
            Set to False for models without chat templates. Default is True.
        model_kwargs: Additional keyword arguments for model initialization (Will be provided as --key=value to the model)
        server_log_folder: Optional directory path where server logs will be stored.
            If provided, creates one log file per rank (e.g., server_rank_0.log). If None, server output
            is muted after startup completion.
        external_endpoint: URL of external inference server (required for "vllm-remote" server_type).
            Example: "http://my-vllm-server.com:8000" or "https://api.service.com"
    """

    server_type: Literal["sglang", "vllm", "dummy", "vllm-remote"]
    model_name_or_path: str
    temperature: float = 0.0
    model_max_context: int = 8192
    max_concurrent_requests: int = 500
    metric_interval: int = 120
    tp: int = 1
    dp: int = 1
    pp: int = 1
    max_concurrent_tasks: int | None = None
    use_chat: bool = True
    model_kwargs: dict | None = None
    server_log_folder: str | None = None
    external_endpoint: str | None = None

    def __post_init__(self):
        if self.max_concurrent_tasks is None:
            self.max_concurrent_tasks = self.max_concurrent_requests


# --------------------------------------------------------------------------- #
# Manages output saving, checkpointing, and chunking
# --------------------------------------------------------------------------- #
class CheckpointManager:
    def __init__(self, checkpoints_local_dir: str | None = None, records_per_chunk: int = 6000):
        """
        Manages checkpointing and chunking of documents using a Queue-based approach.

        If checkpoints_local_dir is provided, it will save documents to it in chunks of records_per_chunk documents.
        If it's not provided, it will only write to the main output writer.

        The Queue-based approach eliminates race conditions by having a single writer task
        that processes all writes sequentially.
        """
        self.checkpoints_local_dir = checkpoints_local_dir if checkpoints_local_dir is not None else None
        self.checkpoints_local_dir_df = (
            get_datafolder(checkpoints_local_dir) if checkpoints_local_dir is not None else None
        )
        if self.checkpoints_local_dir_df is not None and not self.checkpoints_local_dir_df.is_local():
            raise ValueError("checkpoints_local_dir must be a local directory")
        if records_per_chunk <= 0:
            raise ValueError("records_per_chunk must be positive")
        self.records_per_chunk = records_per_chunk

        # Queue-based approach (no locks needed!)
        self.write_queue: asyncio.Queue | None = None
        self.writer_task: asyncio.Task | None = None

        self.checkpoint_file_lock = asyncio.Lock()
        self.per_chunk_counts = Counter()
        self.new_completed_chunks = set()
        self.last_chunk_index = -1

    async def start_writer(self, queue_size: int = 10000):
        """Start the checkpoint writer task."""
        if self.write_queue is not None:
            logger.warning("Writer task already started")
            return

        self.write_queue = asyncio.Queue(maxsize=queue_size)
        self.writer_task = asyncio.create_task(self.checkpoint_writer_task())
        logger.info("Checkpoint writer task started")

    async def stop_writer(self):
        """Stop the checkpoint writer task gracefully."""
        if self.write_queue is None:
            return

        # Send shutdown signal
        await self.write_queue.put(None)
        # Wait for queue to drain
        await self.write_queue.join()
        # Wait for writer task to finish
        if self.writer_task is not None:
            await self.writer_task
        logger.info("Checkpoint writer task stopped")

    async def checkpoint_writer_task(self):
        """
        Single writer task - processes all checkpoint writes sequentially.

        This eliminates race conditions by design. Only one task writes to files,
        so there's no concurrent access to shared state.
        """
        import orjson

        while True:
            item = await self.write_queue.get()

            if item is None:  # Shutdown signal
                self.write_queue.task_done()
                break

            try:
                document, rank, chunk_index, output_writer_context = item

                # Write to main output writer (ParquetWriter has its own lock)
                if "postprocess_remove" not in document.metadata:
                    output_writer_context.write(document, rank=rank, chunk_index=chunk_index)

                self.per_chunk_counts[chunk_index] += 1

                # Write to checkpoint JSONL
                if self.checkpoints_local_dir is not None:
                    save_path = os.path.join(
                        self.checkpoints_local_dir,
                        f"{rank:05d}/chunk_{chunk_index:05d}.jsonl"
                    )
                    save_dir = os.path.dirname(save_path)

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)

                    file_exists = os.path.exists(save_path)
                    if not file_exists:
                        logger.info(f"Creating checkpoint file {save_path}")

                    # Synchronous file write (no aiofiles, no race)
                    with open(save_path, "ab") as f:
                        f.write(orjson.dumps(dataclasses.asdict(document), option=orjson.OPT_APPEND_NEWLINE))
                        f.flush()
                        os.fsync(f.fileno())  # Force to disk

                # Check if chunk complete
                if self.per_chunk_counts[chunk_index] == self.records_per_chunk:
                    filename = output_writer_context._get_output_filename(
                        document, rank, chunk_index=chunk_index
                    )
                    output_writer_context.close_file(filename)
                    self.new_completed_chunks.add(chunk_index)
                    # Update last chunk index will be called separately

            except Exception as e:
                logger.error(f"Error in checkpoint writer: {e}", exc_info=True)
            finally:
                self.write_queue.task_done()

    async def write_document(self, document: Document, rank: int, chunk_index: int, output_writer_context: DiskWriter):
        """
        Queue-based write - just enqueue, no locks needed.

        The single writer task processes all writes sequentially, eliminating race conditions.
        """
        if self.write_queue is None:
            raise RuntimeError("CheckpointManager not started. Call start_writer() first.")

        # Simply enqueue - writer task handles everything
        await self.write_queue.put((document, rank, chunk_index, output_writer_context))

    async def parse_existing_checkpoints(self, rank: int, output_writer_context: DiskWriter) -> tuple[int, set[str]]:
        """
        Load all checkpoints for a given rank and write them to the output writer.
        Returns:
        - documents to skip: number of documents from completed chunks that were already finished
        - set of ids of documents that were already processed in the unfinished chunks
        """
        all_ids = set()
        if not self.checkpoints_local_dir:
            return 0, all_ids

        async with self.checkpoint_file_lock:
            if self.checkpoints_local_dir_df.exists(f"last_chunk/{rank:05d}.txt"):
                with self.checkpoints_local_dir_df.open(f"last_chunk/{rank:05d}.txt", "r") as f:
                    self.last_chunk_index = int(f.read().strip())

            reader = JsonlReader(self.checkpoints_local_dir, compression=None)
            should_update_last_chunk_index = False
            # find existing chunk files and read from them
            for filename in self.checkpoints_local_dir_df.glob(f"{rank:05d}/*.jsonl"):
                chunk_index = int(filename.removeprefix(f"{rank:05d}/chunk_").removesuffix(".jsonl"))
                # Queue-based approach: No per-chunk locks needed (single writer task)
                for document in reader.read_file(filename):
                    if "postprocess_remove" not in document.metadata:
                        output_writer_context.write(document, rank=rank, chunk_index=chunk_index)
                    all_ids.add(document.id)
                    self.per_chunk_counts[chunk_index] += 1
                    if self.per_chunk_counts[chunk_index] == self.records_per_chunk:
                        # close the file
                        filename = output_writer_context._get_output_filename(
                            document, rank, chunk_index=chunk_index
                        )
                        # Use asyncio.to_thread to avoid blocking event loop with I/O
                        await asyncio.to_thread(output_writer_context.close_file, filename)
                        self.new_completed_chunks.add(chunk_index)
                        # update the last chunk index/delete local file etc
                        should_update_last_chunk_index = True
        # can not be within the checkpoint_file_lock - call outside to avoid deadlock
        if should_update_last_chunk_index:
            await self.update_last_chunk_index(rank)
        return (self.last_chunk_index + 1) * self.records_per_chunk if self.last_chunk_index >= 0 else 0, all_ids

    async def cleanup_last_chunk(self, rank: int, chunk_index: int):
        import shutil

        def remove_rank_dir(rank_dir: str, chunk_index: int):
            """Helper to remove rank directory if conditions are met"""
            if os.path.exists(rank_dir) and self.last_chunk_index == chunk_index:
                shutil.rmtree(rank_dir)

        if self.checkpoints_local_dir is not None:
            self.new_completed_chunks.add(chunk_index)
            await self.update_last_chunk_index(rank)
            rank_dir = os.path.join(self.checkpoints_local_dir, f"{rank:05d}")
            # second part should be redundant as we technically only call this after everything completes but seems buggy for now
            # Use asyncio.to_thread to avoid blocking event loop with I/O
            await asyncio.to_thread(remove_rank_dir, rank_dir, chunk_index)

    async def update_last_chunk_index(self, rank: int):
        """
        Update the last chunk index and delete the local file if it's complete.
        """
        import os

        def remove_chunk_file(chunk_file: str):
            """Helper to remove chunk file if it exists"""
            if os.path.exists(chunk_file):
                os.remove(chunk_file)

        def write_last_chunk_index(filepath: str, content: str):
            """Helper to write last chunk index"""
            with self.checkpoints_local_dir_df.open(filepath, "wt") as f:
                f.write(content)

        async with self.checkpoint_file_lock:
            # possibly multiple ones, in case file +2 finished before +1
            while self.last_chunk_index + 1 in self.new_completed_chunks:
                self.last_chunk_index += 1
                chunk_file = os.path.join(
                    self.checkpoints_local_dir, f"{rank:05d}/chunk_{self.last_chunk_index:05d}.jsonl"
                )
                # Use asyncio.to_thread to avoid blocking event loop with I/O
                await asyncio.to_thread(remove_chunk_file, chunk_file)
                logger.info(f"Finished chunk {self.last_chunk_index}")
                # clean up - use pop with default to avoid KeyError if chunk wasn't tracked
                self.per_chunk_counts.pop(self.last_chunk_index, None)
                self.new_completed_chunks.remove(self.last_chunk_index)
                # save new last chunk index
                # Use asyncio.to_thread to avoid blocking event loop with I/O
                await asyncio.to_thread(
                    write_last_chunk_index, f"last_chunk/{rank:05d}.txt", str(self.last_chunk_index)
                )

    def chunk_index_gen(self):
        ci = 0
        while True:
            for _ in range(self.records_per_chunk):
                yield ci
            ci += 1


# --------------------------------------------------------------------------- #
# Minimal inference runner
# --------------------------------------------------------------------------- #
class InferenceRunner(PipelineStep):
    """
    Pipeline step for running inference on documents using various inference servers.

    This runner pulls documents from readers, converts them to LLM requests via a query builder,
    sends requests to a locally spawned inference server, and processes the responses through
    post-processing steps.

    Inference results are saved in document metadata as "inference_results" list.
    Each inference result is either InferenceSuccess or InferenceError.
    """

    name = "Inference 🔍"
    type = "Model call"

    def __init__(
        self,
        query_builder: Callable[[InferenceRunner, Document], AsyncGenerator[dict, None] | dict],
        config: InferenceConfig,
        output_writer: DiskWriter,
        checkpoints_local_dir: str | None = None,
        records_per_chunk: int = 6000,
        postprocess_fn: Callable[[InferenceRunner, Document], Document | None] | None = None,
        skip_bad_requests: bool = False,
        max_concurrent_scoring: int = 50,
    ):
        """
        Initialize the inference runner.

        Args:
            query_builder: Function that returns inference request payload(s) for a document.
                          Can return either:
                          - AsyncGenerator[dict, None]: async generator yielding dicts
                          - dict: single payload dict
            config: Configuration for the inference server and processing
            output_writer: Writer for saving inference results
            checkpoints_local_dir: Local directory to store checkpoints. We save individual files of records_per_chunk documents each locally as a "copy" of the output_writer documents. If a task fails, we will take the locally saved files and re-upload their documents.
            records_per_chunk: Ignored if checkpoints_local_dir is not provided. Default: 6000.
            skip_bad_requests: If True, will skip documents that cause BadRequestError from the server. Default: False.
            postprocess_fn: Function that post-processes the document after inference. Takes the InferenceRunner instance and document as arguments. If it returns None, the document is not saved to output_writer. Can be either sync or async function.
            max_concurrent_scoring: Maximum number of concurrent scoring requests across all documents. Used when postprocess_fn is async to limit concurrent scoring operations. Default: 50.
        """
        super().__init__()

        self.query_builder = query_builder
        self.config = config
        self.postprocess_fn = postprocess_fn
        self.skip_bad_requests = skip_bad_requests
        self.max_concurrent_scoring = max_concurrent_scoring

        self.output_writer = output_writer

        self.checkpoint_manager = CheckpointManager(checkpoints_local_dir, records_per_chunk)

        self._server: InferenceServer | None = None
        self.metrics = MetricsKeeper(window=60 * 5)
        self.queue_sizes = QueueSizesKeeper()

        # Scoring semaphore will be initialized in run() where event loop exists
        self._scoring_semaphore: asyncio.Semaphore | None = None

        # Queue for yielding documents in pipeline chaining mode
        # When not None, processed documents are queued for yielding to next pipeline step
        self._processed_documents_queue: asyncio.Queue | None = None

    async def metrics_reporter(self, interval: int = 600):
        """
        Periodically report metrics and queue sizes.

        Args:
            interval: Reporting interval in seconds
        """
        while True:
            # Leading newlines preserve table formatting in logs
            logger.info("\n" + str(self.metrics))
            logger.info("\n" + str(self.queue_sizes))
            logger.info(str(self.stats))
            await asyncio.sleep(interval)

    @property
    def scoring_semaphore(self) -> asyncio.Semaphore:
        """
        Access scoring semaphore for limiting concurrent scoring operations.

        Returns:
            The scoring semaphore instance

        Raises:
            RuntimeError: If called before run_async initializes the semaphore
        """
        if self._scoring_semaphore is None:
            raise RuntimeError("scoring_semaphore accessed before run_async initialization")
        return self._scoring_semaphore

    @property
    def server(self) -> InferenceServer:
        """
        Lazy initialization of the inference server.

        Returns:
            The initialized inference server instance
        """
        if self._server is None:
            self._server = self._init_server()
        # At this point _server is guaranteed to be not None after _init_server()
        assert self._server is not None
        return self._server

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _init_server(self) -> InferenceServer:
        """
        Initialize the requested inference server (non-blocking).

        For local servers (vllm, sglang, dummy), spawns a new server process.
        For remote servers (vllm-remote), connects to an existing external endpoint.

        Returns:
            The initialized inference server instance

        Raises:
            ValueError: If unsupported server type is specified
        """
        stype = self.config.server_type

        # Local servers
        if stype == "sglang":
            return SGLangServer(self.config)
        elif stype == "vllm":
            return VLLMServer(self.config)
        elif stype == "dummy":
            return DummyServer(self.config)

        # Remote servers
        elif stype == "vllm-remote":
            return VLLMRemoteServer(self.config)

        else:
            raise ValueError(f"Unsupported server type: {stype}")

    async def _send_request(self, payload: dict, semaphore: asyncio.Semaphore) -> InferenceSuccess | InferenceError:
        """
        POST payload to the local server and return the parsed result.

        Args:
            payload: The request payload to send
            semaphore: Semaphore for controlling concurrent requests

        Returns:
            InferenceSuccess with response data or InferenceError with error message
        """
        # Choose endpoint based on use_chat setting
        if self.config.use_chat:
            endpoint = "/v1/chat/completions"
        else:
            endpoint = "/v1/completions"

        url = f"http://localhost:{self.server.port}{endpoint}"
        max_retries = 6
        attempt = 0

        self.queue_sizes.change_queues({"waiting_requests": 1})
        async with semaphore:
            self.queue_sizes.change_queues({"waiting_requests": -1})
            self.queue_sizes.change_queues({"running_requests": 1})

            while attempt < max_retries:
                try:
                    status, body = await _raw_post(url, json_data=payload)
                    if status == 400:
                        self.queue_sizes.change_queues({"running_requests": -1})
                        return InferenceError(error=f"Got BadRequestError from server: {body.decode()}")
                    elif status == 500:
                        self.queue_sizes.change_queues({"running_requests": -1})
                        return InferenceError(error=f"Got InternalServerError from server: {body.decode()}")
                    elif status != 200:
                        self.queue_sizes.change_queues({"running_requests": -1})
                        return InferenceError(error=f"Error http status {status}")

                    response = json.loads(body)
                    choice = response["choices"][0]

                    # Track metrics
                    usage = response.get("usage", {})
                    self.metrics.add_metrics(
                        tokens_input=usage.get("prompt_tokens", 0),
                        tokens_output=usage.get("completion_tokens", 0),
                    )

                    # Parse response based on endpoint type
                    if self.config.use_chat:
                        # Reconstruct GPT OSS format if vLLM reasoning parser was used
                        text = reconstruct_gptoss_from_vllm_response(choice)
                    else:
                        text = choice["text"]

                    self.queue_sizes.change_queues({"running_requests": -1})
                    return InferenceSuccess(text=text, finish_reason=choice["finish_reason"], usage=usage)
                except (ConnectionError, OSError, asyncio.TimeoutError) as e:
                    # This means the server is dead likely, so we need to wait for restart
                    logger.warning(f"Client error: {type(e)} {e}")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    sleep_delay = 5 * (2**attempt)
                    await asyncio.sleep(sleep_delay)
                    attempt += 1
                except asyncio.CancelledError:
                    logger.info("Request cancelled")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    raise
                except Exception as e:
                    logger.warning(f"Unexpected error: {type(e)} {e}")
                    self.queue_sizes.change_queues({"running_requests": -1})
                    return InferenceError(error=str(e))

            self.queue_sizes.change_queues({"running_requests": -1})
            return InferenceError(error=f"Failed to process request after {max_retries} attempts")

    async def _save_document(self, document: Document, output_writer_context: DiskWriter, rank: int, chunk_index: int):
        """
        Save processed document to results queue.

        Args:
            document: The processed document to save
            output_writer_context: Context manager for the output writer
            rank: Process rank identifier
            chunk_index: Chunk index to save the document to
        """
        # Track document metrics
        try:
            inference_results = document.metadata.get("inference_results", [])  # type: ignore
            successful_requests = sum(1 for result in inference_results if isinstance(result, InferenceSuccess))  # type: ignore
            failed_requests = len(inference_results) - successful_requests  # type: ignore

            # Track tokens for each inference result
            total_input_tokens = 0
            total_output_tokens = 0
            for result in inference_results:
                if isinstance(result, InferenceSuccess):
                    prompt_tokens = result.usage.get("prompt_tokens", 0)  # type: ignore
                    completion_tokens = result.usage.get("completion_tokens", 0)  # type: ignore
                    total_input_tokens += prompt_tokens
                    total_output_tokens += completion_tokens

                    # Update stats for each individual request
                    self.stat_update("prompt_tokens", value=prompt_tokens, unit="request")
                    self.stat_update("completion_tokens", value=completion_tokens, unit="request")

            self.metrics.add_metrics(
                tokens_finished_input=total_input_tokens,
                tokens_finished_output=total_output_tokens,
                requests=len(inference_results),  # type: ignore
            )

            self.stat_update("successful_requests", value=successful_requests, unit="document")
            self.stat_update("failed_requests", value=failed_requests, unit="document")
            self.stat_update("successful_documents", value=1)

            await self.checkpoint_manager.write_document(document, rank, chunk_index, output_writer_context)

            # Queue document for yielding if in pipeline chain mode
            if self._processed_documents_queue is not None:
                await self._processed_documents_queue.put(document)

        except Exception as e:
            logger.warning(f"Failed to process inference results for metrics: {e}")
            self.stat_update("failed_documents", value=1)

    async def _async_data_gen(self, sync_gen: Iterable[Document]):
        """
        Convert synchronous generator to async generator using asyncio.to_thread.

        Args:
            sync_gen: Synchronous iterable of documents

        Yields:
            Document objects from the synchronous generator
        """

        def get_next_item(iterator):
            try:
                return next(iterator), False
            except StopIteration:
                return None, True

        iterator = iter(sync_gen)
        while True:
            item, is_done = await asyncio.to_thread(get_next_item, iterator)
            if is_done:
                break
            yield item

    # --------------------------------------------------------------------- #
    # Async processing
    # --------------------------------------------------------------------- #
    async def run_async(
        self,
        data_gen: Iterable[Document],
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Run asynchronous inference processing on the provided data.

        Args:
            data_gen: Iterable of Document objects to process
            rank: Process rank identifier for distributed processing
            world_size: Total number of processes in distributed setup
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._scoring_semaphore = asyncio.Semaphore(self.max_concurrent_scoring)
        server_task = asyncio.create_task(self.server.host_server(rank=rank))
        await self.server.wait_until_ready()
        logger.info(f"Inference server up on port {self.server.port}")

        # Start checkpoint writer task
        await self.checkpoint_manager.start_writer()

        # Start metrics reporting
        self.metrics.reset()
        metrics_task = asyncio.create_task(self.metrics_reporter(interval=self.config.metric_interval))

        async def _handle_record(
            doc: Document, rank: int, chunk_index: int, output_writer_context: DiskWriter
        ) -> None:
            """
            Process a single document through the inference pipeline.

            Args:
                doc: Document to process
                rank: Process rank identifier
                chunk_index: Chunk index for the document
                output_writer_context: Output writer context for saving documents

            Raises:
                InferenceProcessingError: If document processing fails
            """
            try:
                # Get payloads from query_builder
                payloads_result = self.query_builder(self, doc)

                # If calling the query_builder returned a coroutine, await it first
                # (happens when query_builder is a callable class with async __call__)
                if asyncio.iscoroutine(payloads_result):
                    payloads_result = await payloads_result

                # Handle different return types
                request_tasks = []

                # Check if it's an async generator
                if isinstance(payloads_result, AsyncGenerator):
                    # It's an async generator - process each payload as soon as it's yielded
                    async for payload in payloads_result:
                        # Set default values for payload
                        payload.setdefault("model", self.config.model_name_or_path)
                        payload.setdefault("temperature", self.config.temperature)

                        # Start request immediately
                        task = asyncio.create_task(self._send_request(payload, semaphore))
                        request_tasks.append(task)

                elif isinstance(payloads_result, dict):
                    # Single dict
                    payload = payloads_result
                    payload.setdefault("model", self.config.model_name_or_path)
                    payload.setdefault("temperature", self.config.temperature)
                    task = asyncio.create_task(self._send_request(payload, semaphore))
                    request_tasks.append(task)

                if not request_tasks:
                    raise InferenceProcessingError(doc, "No valid payloads generated from query_builder")

                # Wait for all requests to complete and collect results in order
                results = await asyncio.gather(*request_tasks)

                for result in results:
                    if isinstance(result, InferenceError) and (
                        not self.skip_bad_requests or "BadRequestError" not in result.error
                    ):
                        # re-raise any non-skippable errors
                        raise InferenceProcessingError(doc, result.error)

                # Store results directly in document metadata
                doc.metadata["inference_results"] = results

                # Post-process the document if a function is provided. We still want the actual document for checkpointing purposes.
                if self.postprocess_fn:
                    # Call the postprocess function (works for both functions and callable classes)
                    result = self.postprocess_fn(self, doc)

                    # Check if the RESULT is a coroutine (works for both async functions and async __call__)
                    if asyncio.iscoroutine(result):
                        postprocess_result = await result
                    else:
                        postprocess_result = result

                    if postprocess_result is None:
                        doc.metadata["postprocess_remove"] = True
                    else:
                        doc = postprocess_result

                await self._save_document(doc, output_writer_context, rank, chunk_index)
            except InferenceProcessingError as e:
                raise e
            except Exception as e:
                # let's propagate it
                raise InferenceProcessingError(doc, e)

        # 2. Main processing loop
        tasks_pool: set[asyncio.Task] = set()
        with self.output_writer as output_writer_context:
            # this will also upload locally cached documents to the output writer
            documents_to_skip, processed_ids = await self.checkpoint_manager.parse_existing_checkpoints(
                rank, output_writer_context
            )
            if documents_to_skip > 0:
                logger.info(
                    f"Resuming from previous checkpoint. Will skip {documents_to_skip + len(processed_ids)} already processed documents"
                )

            # process remaining documents
            record_idx = -1
            chunk_index = -1  # Initialize to handle empty input (no documents processed)
            chunk_index_gen = self.checkpoint_manager.chunk_index_gen()
            async for record in self._async_data_gen(data_gen):
                record_idx += 1
                chunk_index = next(chunk_index_gen)
                # Skip documents if resuming from checkpoint
                if record_idx < documents_to_skip:
                    continue
                elif record_idx == documents_to_skip and documents_to_skip > 0:
                    logger.info(f"Skipped {documents_to_skip} documents. Resuming from chunk {chunk_index}")

                # skip already processed documents from chunks in progress
                if record.id in processed_ids:
                    processed_ids.remove(record.id)
                    continue

                # Throttle by task pool size
                while len(tasks_pool) >= self.config.max_concurrent_tasks:
                    done, tasks_pool = await asyncio.wait(tasks_pool, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        await task  # Re-raises any unhandled exception

                # Add task for current record
                task = asyncio.create_task(_handle_record(record, rank, chunk_index, output_writer_context))
                tasks_pool.add(task)

            # 3. Wait for all remaining tasks to complete
            if tasks_pool:
                await asyncio.gather(*tasks_pool)

            # Stop checkpoint writer and wait for queue to drain
            await self.checkpoint_manager.stop_writer()

            # Cleanup after writer stopped - only if we processed documents
            if record_idx >= 0:  # Guard against empty input (no documents processed)
                await self.checkpoint_manager.cleanup_last_chunk(rank, chunk_index)

            # 4. Close any incomplete chunks before context exit
            # This ensures Parquet files are finalized even if records_per_chunk wasn't reached
            # Queue writer is stopped, safe to call close_file directly
            for chunk_idx, count in self.checkpoint_manager.per_chunk_counts.items():
                if count > 0 and count < self.checkpoint_manager.records_per_chunk:
                    # Incomplete chunk - need explicit close to finalize Parquet files
                    from datatrove.data import Document
                    dummy = Document(text="", id="", metadata={})
                    filename = output_writer_context._get_output_filename(
                        dummy, rank, chunk_index=chunk_idx
                    )
                    # Call synchronously (queue writer already stopped)
                    output_writer_context.close_file(filename)

        # 5. Signal completion if in pipeline chain mode
        if self._processed_documents_queue is not None:
            await self._processed_documents_queue.put(None)  # Sentinel value to signal completion

        # 6. shutdown inference server and metrics
        server_task.cancel()
        metrics_task.cancel()

    # --------------------------------------------------------------------- #
    # Synchronous entrypoint required by PipelineStep
    # --------------------------------------------------------------------- #
    def run(
        self,
        data: Iterable[Document],
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Consume `data`, run inference and post-processing, do not yield further documents.

        Args:
            data: Iterable of Document objects to process
            rank: Process rank identifier for distributed processing
            world_size: Total number of processes in distributed setup
        """
        with self.track_time(unit="total"):
            asyncio.run(self.run_async(data, rank, world_size))

    def run_with_yield(
        self,
        data: Iterable[Document],
        rank: int = 0,
        world_size: int = 1,
    ) -> DocumentsPipeline:
        """
        Process `data` with inference and yield processed documents for pipeline chaining.

        This method enables InferenceRunner to be used as an intermediate pipeline step,
        allowing downstream steps (like stats collection) to consume the processed documents.

        Args:
            data: Iterable of Document objects to process
            rank: Process rank identifier for distributed processing
            world_size: Total number of processes in distributed setup

        Yields:
            Processed Document objects with inference results
        """
        import asyncio
        import concurrent.futures
        from queue import Queue, Empty

        # Create sync queue for thread-safe communication between async and sync contexts
        sync_queue: Queue[Document | None] = Queue()

        async def async_worker():
            """Run async processing and populate queue."""
            # Initialize async queue for internal use
            self._processed_documents_queue = asyncio.Queue()

            # Launch async processing in background
            async_task = asyncio.create_task(self.run_async(data, rank, world_size))

            # Transfer documents from async queue to sync queue
            try:
                while True:
                    doc = await self._processed_documents_queue.get()
                    sync_queue.put(doc)  # Put into sync queue for yielding
                    if doc is None:  # Sentinel value signals completion
                        break
            except Exception as e:
                logger.error(f"Error in async worker: {e}")
                sync_queue.put(None)  # Ensure sentinel is sent even on error
                raise
            finally:
                # Wait for async processing to complete
                await async_task
                # Clean up
                self._processed_documents_queue = None

        # Run async worker in background thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(async_worker()))

            # Yield documents from sync queue
            with self.track_time(unit="total"):
                while True:
                    try:
                        # Get document from queue (timeout to allow checking for exceptions)
                        doc = sync_queue.get(timeout=0.1)

                        if doc is None:  # Sentinel value signals completion
                            break

                        yield doc

                    except Empty:
                        # Check if background task failed
                        if future.done():
                            # Re-raise any exception from async worker
                            future.result()
                            break
                        continue

            # Wait for background task to complete
            future.result()
