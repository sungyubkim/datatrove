"""Formatter to transform RLVR-IFeval data to IFBench-VERL format.

This module provides a PipelineStep that converts RLVR-IFeval dataset examples
into the IFBench-VERL format for use with datatrove's IFEval scorer.
"""

from typing import TYPE_CHECKING

from datatrove.pipeline.base import PipelineStep

if TYPE_CHECKING:
    from datatrove.data import Document


class RLVRToIFBenchFormatter(PipelineStep):
    """Transform RLVR-IFeval examples to IFBench-VERL format.

    This formatter converts examples from the RLVR-IFeval dataset format
    to the IFBench-VERL format, enabling scoring with datatrove's IFEval scorer.

    The transformation includes:
    - Mapping RLVR function names to IFEval instruction IDs
    - Converting parameter names (e.g., N → num_paragraphs)
    - Normalizing quantifiers (e.g., "at most N" → "less than N+1")
    - Formatting ground truth as Python literals (using repr())

    Args:
        None

    Example:
        >>> from datatrove.pipeline.readers import HuggingFaceReader
        >>> from datatrove.pipeline.formatters import RLVRToIFBenchFormatter
        >>> from datatrove.pipeline.writers import JsonlWriter
        >>>
        >>> reader = HuggingFaceReader("allenai/RLVR-IFeval", split="train")
        >>> formatter = RLVRToIFBenchFormatter()
        >>> writer = JsonlWriter("output/ifbench-rlvr")
        >>>
        >>> pipeline = reader | formatter | writer
    """

    name = "⚙️ RLVR→IFBench"

    def __init__(self):
        """Initialize the formatter."""
        super().__init__()

    def run(self, data, rank: int = 0, world_size: int = 1):
        """Transform RLVR examples to IFBench-VERL format.

        Args:
            data: Generator of Documents containing RLVR examples
            rank: Rank of this task
            world_size: Total number of tasks

        Yields:
            Documents in IFBench-VERL format
        """
        from datatrove.data import Document
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench

        for idx, doc in enumerate(data):
            with self.track_time():
                try:
                    # Parse the original RLVR data from document metadata
                    rlvr_example = {
                        "messages": doc.metadata.get("messages"),
                        "ground_truth": doc.metadata.get("ground_truth"),
                        "dataset": doc.metadata.get("dataset", "ifeval"),
                    }

                    # Transform to IFBench format
                    # Use global index: rank * batch_size + local_idx
                    global_idx = rank * 10000 + idx  # Simple sharding assumption
                    ifbench_example = transform_to_ifbench(rlvr_example, index=global_idx)

                    # Create new document with transformed data
                    # Store the entire ifbench structure in metadata
                    new_doc = Document(
                        text=doc.text,  # Preserve original text if any
                        id=f"ifbench-rlvr-{global_idx}",
                        metadata=ifbench_example,
                    )

                    self.stat_update("transformed")
                    yield new_doc

                except Exception as e:
                    self.logger.warning(f"Failed to transform document {doc.id}: {e}")
                    self.stat_update("failed")
                    continue
