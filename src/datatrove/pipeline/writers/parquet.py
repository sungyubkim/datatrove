from collections import Counter, defaultdict
import threading
from typing import IO, Any, Callable, Literal

from datatrove.io import DataFolderLike
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.logging import logger


class ParquetWriter(DiskWriter):
    default_output_filename: str = "${rank}.parquet"
    name = "📒 Parquet"
    _requires_dependencies = ["pyarrow"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = "snappy",
        adapter: Callable = None,
        batch_size: int = 1000,
        expand_metadata: bool = False,
        max_file_size: int = 5 * 2**30,  # 5GB
        schema: Any = None,
    ):
        # Validate the compression setting
        if compression not in {"snappy", "gzip", "brotli", "lz4", "zstd", None}:
            raise ValueError(
                "Invalid compression type. Allowed types are 'snappy', 'gzip', 'brotli', 'lz4', 'zstd', or None."
            )

        super().__init__(
            output_folder,
            output_filename,
            compression=None,  # Ensure superclass initializes without compression
            adapter=adapter,
            mode="wb",
            expand_metadata=expand_metadata,
            max_file_size=max_file_size,
        )
        self._writers = {}
        self._batches = defaultdict(list)
        self._file_counter = Counter()
        self.compression = compression
        self.batch_size = batch_size
        self.schema = schema
        # Thread lock to protect _batches and _writers from concurrent access
        self._write_lock = threading.Lock()

    def _on_file_switch(self, original_name, old_filename, new_filename):
        """
            Called when we are switching file from "old_filename" to "new_filename" (original_name is the filename
            without 000_, 001_, etc)
        Args:
            original_name: name without file counter
            old_filename: old full filename
            new_filename: new full filename
        """
        logger.warning(
            f"[PARQUET DEBUG] _on_file_switch() called: "
            f"original_name={original_name}, old={old_filename}, new={new_filename}, "
            f"batch_size={len(self._batches.get(original_name, []))}"
        )

        with self._write_lock:
            # Note: parent class _on_file_switch calls _write_batch to flush current batch
            self._writers.pop(original_name).close()
            super()._on_file_switch(original_name, old_filename, new_filename)

        logger.warning(f"[PARQUET DEBUG] _on_file_switch() completed")

    def close_file(self, original_name: str):
        """
        Override to properly flush batches and close PyArrow writer.

        This ensures that:
        1. Any buffered data in _batches is written to the file
        2. The PyArrow ParquetWriter writes the footer and metadata
        3. The file handler is properly closed

        Args:
            original_name: Logical filename from _get_output_filename()
        """
        # Use lock to ensure thread-safe access to _batches and _writers
        with self._write_lock:
            # Debug: Log all batch states before closing
            batch_summary = {k: len(v) for k, v in self._batches.items()}
            logger.warning(
                f"[PARQUET DEBUG] close_file() called for {original_name}, "
                f"all_batches={batch_summary}"
            )

            try:
                # 1. Flush any remaining batch data for this file
                if original_name in self._batches:
                    batch_size = len(self._batches[original_name])
                    if batch_size > 0:
                        logger.info(f"Flushing {batch_size} remaining documents to {original_name}")
                        self._write_batch(original_name)

                # 2. Close PyArrow ParquetWriter (writes footer/metadata)
                if original_name in self._writers:
                    self._writers.pop(original_name).close()
                    logger.info(f"Successfully closed Parquet file: {original_name}")

                # 3. Close file handler (handles 000_ prefix via parent class)
                super().close_file(original_name)
            except Exception as e:
                logger.error(f"Error closing file {original_name}: {e}", exc_info=True)
                raise

    def _write_batch(self, filename):
        if not self._batches[filename]:
            return
        import pyarrow as pa
        from datatrove.utils.logging import logger

        batch_size = len(self._batches[filename])
        logger.warning(f"[PARQUET DEBUG] _write_batch() called for {filename}, writing {batch_size} documents")

        # prepare batch
        batch = pa.RecordBatch.from_pylist(self._batches.pop(filename), schema=self.schema)
        # write batch
        self._writers[filename].write_batch(batch)

        logger.warning(f"[PARQUET DEBUG] _write_batch() completed for {filename}, {batch_size} documents written to PyArrow")

    def _write(self, document: dict, file_handler: IO, filename: str):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datatrove.utils.logging import logger

        # Use lock to ensure thread-safe access to _batches and _writers
        with self._write_lock:
            if filename not in self._writers:
                self._writers[filename] = pq.ParquetWriter(
                    file_handler,
                    schema=self.schema if self.schema is not None else pa.RecordBatch.from_pylist([document]).schema,
                    compression=self.compression,
                )
                logger.warning(f"[PARQUET DEBUG] Created writer for {filename}, batch_size={self.batch_size}")

            batch_size_before = len(self._batches[filename])
            self._batches[filename].append(document)
            batch_size_after = len(self._batches[filename])

            # Log every 10th append or when close to batch_size
            if batch_size_after % 10 == 0 or batch_size_after >= self.batch_size - 5:
                logger.warning(
                    f"[PARQUET DEBUG] Appended to {filename}: "
                    f"batch_size_before={batch_size_before}, batch_size_after={batch_size_after}, "
                    f"target_batch_size={self.batch_size}"
                )

            if len(self._batches[filename]) == self.batch_size:
                logger.warning(f"[PARQUET DEBUG] AUTO-FLUSH triggered for {filename} (batch_size={self.batch_size})")
                self._write_batch(filename)

    def close(self):
        import traceback

        # Log who called close() and why
        stack = ''.join(traceback.format_stack()[-5:-1])  # Last 4 frames before this one
        logger.warning(
            f"[PARQUET DEBUG] close() CALLED! "
            f"This will clear {len(self._writers)} writers and {len(self._batches)} batches.\n"
            f"Call stack:\n{stack}"
        )

        with self._write_lock:
            # Log state before clearing
            batch_summary = {k: len(v) for k, v in self._batches.items()}
            logger.warning(f"[PARQUET DEBUG] Flushing remaining batches: {batch_summary}")

            for filename in list(self._batches.keys()):
                self._write_batch(filename)
            for writer in self._writers.values():
                writer.close()
            self._batches.clear()
            self._writers.clear()
            super().close()

        logger.warning(f"[PARQUET DEBUG] close() COMPLETED - all writers cleared")
