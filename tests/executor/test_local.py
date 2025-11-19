import os
import shutil
import tempfile
import unittest

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.io import get_datafolder
from datatrove.utils._import_utils import is_boto3_available, is_moto_available, is_s3fs_available

from ..utils import require_boto3, require_moto, require_pyarrow, require_s3fs


EXAMPLE_DIRS = ("/home/testuser/somedir", "file:///home/testuser2/somedir", "s3://test-bucket/somedir")
FULL_PATHS = (
    "/home/testuser/somedir/file.txt",
    "/home/testuser2/somedir/file.txt",
    "s3://test-bucket/somedir/file.txt",
)


port = 5555
endpoint_uri = "http://127.0.0.1:%s/" % port


if is_boto3_available():
    import boto3  # noqa: F811

if is_moto_available():
    from moto.moto_server.threaded_moto_server import ThreadedMotoServer  # noqa: F811

if is_s3fs_available():
    from s3fs import S3FileSystem  # noqa: F811


@require_moto
class TestLocalExecutor(unittest.TestCase):
    def setUp(self):
        self.server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
        self.server.start()
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_ACCESS_KEY_ID"] = "foo"

        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)
        self.addCleanup(self.server.stop)

    @require_boto3
    @require_s3fs
    def test_executor(self):
        s3fs = S3FileSystem(client_kwargs={"endpoint_url": endpoint_uri})
        s3 = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_uri)
        s3.create_bucket(Bucket="test-bucket")
        configurations = (3, 1), (3, 3), (3, -1)
        file_list = [
            "executor.json",
            "stats.json",
        ] + [
            x
            for rank in range(3)
            for x in (f"completions/{rank:05d}", f"logs/task_{rank:05d}.log", f"stats/{rank:05d}.json")
        ]
        for tasks, workers in configurations:
            for log_dir in (f"{self.tmp_dir}/{tasks}_{workers}", (f"s3://test-bucket/logs/{tasks}_{workers}", s3fs)):
                log_dir = get_datafolder(log_dir)
                executor = LocalPipelineExecutor(pipeline=[], tasks=tasks, workers=workers, logging_dir=log_dir)
                executor.run()

                for file in file_list:
                    assert log_dir.isfile(file)


@require_pyarrow
class TestLocalExecutorParquetMultiprocessing(unittest.TestCase):
    """
    Test LocalPipelineExecutor with ParquetWriter in multiprocessing mode.

    This verifies the fix from commit 4048a10 which added pickle support to
    ParquetWriter for multiprocessing contexts.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_parquet_writer_multiprocessing_3_tasks(self):
        """
        Test ParquetWriter works with LocalPipelineExecutor using 3 parallel tasks.

        This is the real-world scenario that the pickling fix enables:
        - LocalPipelineExecutor deepcopies the pipeline for each worker
        - Each worker gets its own ParquetWriter instance
        - All workers write to different files without interference
        """
        from pathlib import Path

        from datatrove.data import Document
        from datatrove.pipeline.readers.parquet import ParquetReader
        from datatrove.pipeline.writers.parquet import ParquetWriter

        # First, create input Parquet files that will be sharded across tasks
        input_dir = Path(self.tmp_dir) / "input"
        input_dir.mkdir()
        output_dir = Path(self.tmp_dir) / "output"
        logs_dir = Path(self.tmp_dir) / "logs"

        # Create 3 input files (each task will process 1 file)
        import pyarrow as pa
        import pyarrow.parquet as pq

        for file_idx in range(3):
            # 10 documents per file = 30 total
            documents = [
                {
                    "text": f"Document {file_idx * 10 + i}",
                    "id": f"doc_{file_idx * 10 + i:05d}",
                    "metadata": {"index": file_idx * 10 + i}
                }
                for i in range(10)
            ]

            table = pa.Table.from_pylist(documents)
            pq.write_table(table, input_dir / f"input_{file_idx}.parquet")

        # Create executor with 3 parallel tasks
        executor = LocalPipelineExecutor(
            pipeline=[
                ParquetReader(str(input_dir)),  # Each task gets different input files
                ParquetWriter(
                    output_folder=str(output_dir),
                    output_filename="${rank}.parquet",
                    compression="snappy",
                    batch_size=3  # Small batch to test flushing
                )
            ],
            tasks=3,  # This triggers deepcopy of ParquetWriter for each worker
            workers=3,
            logging_dir=str(logs_dir)
        )

        # Run - this will deepcopy ParquetWriter 3 times
        executor.run()

        # Verify 3 output files created (one per rank)
        output_files = sorted(output_dir.glob("*.parquet"))
        assert len(output_files) == 3, f"Expected 3 output files, got {len(output_files)}: {[f.name for f in output_files]}"

        # Verify each file is valid Parquet
        total_docs = 0
        all_ids = set()

        for rank, output_file in enumerate(output_files):
            # Check magic bytes (PAR1)
            with open(output_file, "rb") as f:
                header = f.read(4)
                assert header == b"PAR1", f"File {output_file.name} missing magic bytes at start. Got: {header!r}"
                f.seek(-4, 2)
                footer = f.read(4)
                assert footer == b"PAR1", f"File {output_file.name} corrupted (missing footer magic bytes). Got: {footer!r}"

            # Read and verify content
            table = pq.read_table(output_file)
            docs_in_file = len(table)
            total_docs += docs_in_file

            # Verify each document ID is unique across all files
            for row in table.to_pylist():
                doc_id = row['id']
                assert doc_id not in all_ids, f"Duplicate document ID found: {doc_id}"
                all_ids.add(doc_id)

        # Verify all documents were processed exactly once
        assert total_docs == 30, f"Expected 30 total documents, got {total_docs}"
        assert len(all_ids) == 30, f"Expected 30 unique document IDs, got {len(all_ids)}"

        # Verify we can read back all documents correctly
        reader = ParquetReader(str(output_dir))
        read_docs = list(reader())
        assert len(read_docs) == 30

        # Verify data correctness (check a few samples)
        for doc in read_docs[:5]:
            assert doc.text.startswith("Document ")
            assert doc.id.startswith("doc_")
            assert "index" in doc.metadata

    def test_parquet_writer_multiprocessing_varying_tasks(self):
        """
        Test ParquetWriter with different task/worker configurations.

        Verifies robustness across various multiprocessing setups.
        """
        from pathlib import Path

        from datatrove.pipeline.readers.parquet import ParquetReader
        from datatrove.pipeline.writers.parquet import ParquetWriter

        # Test different configurations: (tasks, workers, num_files, docs_per_file)
        configurations = [
            (1, 1, 1, 10),    # Single process (no multiprocessing)
            (2, 2, 2, 10),    # 2 parallel tasks
            (4, 2, 4, 10),    # 4 tasks, 2 workers (sequential batches)
        ]

        for tasks, workers, num_files, docs_per_file in configurations:
            num_docs = num_files * docs_per_file
            with self.subTest(tasks=tasks, workers=workers, num_docs=num_docs):
                import pyarrow as pa
                import pyarrow.parquet as pq

                input_dir = Path(self.tmp_dir) / f"input_{tasks}_{workers}"
                input_dir.mkdir(exist_ok=True)
                output_dir = Path(self.tmp_dir) / f"output_{tasks}_{workers}"
                logs_dir = Path(self.tmp_dir) / f"logs_{tasks}_{workers}"

                # Create input files
                for file_idx in range(num_files):
                    documents = [
                        {
                            "text": f"Doc {file_idx * docs_per_file + i}",
                            "id": f"id_{file_idx * docs_per_file + i}",
                            "metadata": {"idx": file_idx * docs_per_file + i}
                        }
                        for i in range(docs_per_file)
                    ]
                    table = pa.Table.from_pylist(documents)
                    pq.write_table(table, input_dir / f"input_{file_idx}.parquet")

                executor = LocalPipelineExecutor(
                    pipeline=[
                        ParquetReader(str(input_dir)),
                        ParquetWriter(
                            output_folder=str(output_dir),
                            output_filename="${rank}.parquet",
                        )
                    ],
                    tasks=tasks,
                    workers=workers,
                    logging_dir=str(logs_dir)
                )

                executor.run()

                # Verify correct number of output files
                output_files = list(output_dir.glob("*.parquet"))
                assert len(output_files) == tasks, f"Expected {tasks} files, got {len(output_files)}"

                # Verify all files are valid
                total_rows = 0
                for output_file in output_files:
                    table = pq.read_table(output_file)
                    total_rows += len(table)

                    # Check magic bytes
                    with open(output_file, "rb") as f:
                        assert f.read(4) == b"PAR1"

                # Verify all documents processed
                assert total_rows == num_docs, f"Expected {num_docs} rows, got {total_rows}"
