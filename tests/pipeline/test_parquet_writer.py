import copy
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

from datatrove.data import Document
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers.parquet import ParquetWriter

from ..utils import require_pyarrow


@require_pyarrow
class TestParquetWriter(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_write(self):
        data = [
            Document(text=text, id=str(i), metadata={"somedata": 2 * i, "somefloat": i * 0.4, "somestring": "hello"})
            for i, text in enumerate(["hello", "text2", "more text"])
        ]
        with ParquetWriter(output_folder=self.tmp_dir, batch_size=2) as w:
            for doc in data:
                w.write(doc)
        reader = ParquetReader(self.tmp_dir)
        c = 0
        for read_doc, original in zip(reader(), data):
            read_doc.metadata.pop("file_path", None)
            assert read_doc == original
            c += 1
        assert c == len(data)

    def test_parquet_writer_deepcopy(self):
        """
        Test that ParquetWriter can be deepcopied (required for multiprocessing).

        This verifies the fix from commit 4048a10 which added __getstate__ and
        __setstate__ methods to handle unpicklable threading.Lock.
        """
        writer = ParquetWriter(
            output_folder=self.tmp_dir,
            output_filename="${rank}.parquet",
            compression="snappy",
            batch_size=5
        )

        # Should not raise exception
        copied_writer = copy.deepcopy(writer)

        # Verify it's a different object
        assert copied_writer is not writer
        assert copied_writer._write_lock is not writer._write_lock

    def test_parquet_writer_pickle(self):
        """
        Test that ParquetWriter can be pickled and unpickled.

        Lower-level test than deepcopy to verify pickle protocol directly.
        """
        writer = ParquetWriter(
            output_folder=self.tmp_dir,
            compression="gzip",
            batch_size=10
        )

        # Pickle and unpickle
        pickled = pickle.dumps(writer)
        unpickled_writer = pickle.loads(pickled)

        # Verify configuration preserved
        assert unpickled_writer.compression == "gzip"
        assert unpickled_writer.batch_size == 10
        assert unpickled_writer.output_folder == writer.output_folder

    def test_parquet_writer_pickling_preserves_config(self):
        """
        Test that all ParquetWriter configuration is preserved after pickling.
        """
        import pyarrow as pa

        schema = pa.schema([
            ('text', pa.string()),
            ('id', pa.string()),
        ])

        writer = ParquetWriter(
            output_folder=self.tmp_dir,
            output_filename="test_${rank}_${id}.parquet",
            compression="zstd",
            batch_size=42,
            schema=schema,
            max_file_size=1000000
        )

        copied = copy.deepcopy(writer)

        # Verify all configuration preserved
        assert copied.output_folder == writer.output_folder
        # Compare template strings, not Template objects (which compare by identity)
        assert copied.output_filename.template == writer.output_filename.template
        assert copied.compression == writer.compression
        assert copied.batch_size == writer.batch_size
        assert copied.schema == writer.schema
        assert copied.max_file_size == writer.max_file_size

        # Verify runtime state starts fresh
        assert len(copied._batches) == 0
        assert len(copied._writers) == 0
        assert copied._write_lock is not None

    def test_parquet_writer_unpickled_functionality(self):
        """
        Test that a deepcopied ParquetWriter actually works correctly.

        This is the critical test: verifying that after deepcopy (used by
        LocalPipelineExecutor for multiprocessing), the writer can still
        write valid Parquet files.
        """
        original_writer = ParquetWriter(
            output_folder=self.tmp_dir,
            output_filename="${rank}.parquet",
            compression="snappy",
            batch_size=3
        )

        # Deepcopy (simulates what LocalPipelineExecutor does)
        copied_writer = copy.deepcopy(original_writer)

        # Use the copied writer to write documents
        data = [
            Document(text=f"Document {i}", id=f"doc_{i}", metadata={"index": i})
            for i in range(10)
        ]

        with copied_writer:
            for doc in data:
                copied_writer.write(doc, rank=0)

        # Verify file was created
        output_files = list(Path(self.tmp_dir).glob("*.parquet"))
        assert len(output_files) == 1, f"Expected 1 file, got {len(output_files)}"

        # Verify file is valid Parquet (check magic bytes)
        output_file = output_files[0]
        with open(output_file, "rb") as f:
            header = f.read(4)
            assert header == b"PAR1", f"Missing magic bytes at start. Got: {header!r}"
            f.seek(-4, 2)
            footer = f.read(4)
            assert footer == b"PAR1", f"Missing magic bytes at end. Got: {footer!r}"

        # Verify content can be read back
        import pyarrow.parquet as pq
        table = pq.read_table(output_file)
        assert len(table) == 10, f"Expected 10 rows, got {len(table)}"

        # Verify data correctness
        reader = ParquetReader(self.tmp_dir)
        read_docs = list(reader())
        assert len(read_docs) == 10
        for read_doc, original_doc in zip(read_docs, data):
            read_doc.metadata.pop("file_path", None)
            assert read_doc.text == original_doc.text
            assert read_doc.id == original_doc.id
