# Parquet File Corruption Issue Analysis

## 📋 Executive Summary

InferenceRunner with ParquetWriter produces corrupted Parquet files (missing magic bytes, 0 bytes size) when using checkpointing. This affects all intermediate chunks during pipeline execution and some files even after completion.

---

## 🔍 Discovered Issues

### Issue 1: Incomplete File Closure in CheckpointManager

**Location:** `src/datatrove/pipeline/inference/run_inference.py:247-251`

**Code:**
```python
if self.per_chunk_counts[chunk_index] == self.records_per_chunk:
    # we gotta close the main file
    output_writer_context.output_mg.pop(
        output_writer_context._get_output_filename(document, rank, chunk_index=chunk_index)
    ).close()
```

**Problem:**
- Only closes the low-level file handler (file descriptor)
- Does NOT flush ParquetWriter's internal `_batches` buffer
- Does NOT properly close ParquetWriter's `_writers` (PyArrow ParquetWriter objects)
- Results in files without Parquet footer/metadata/magic bytes

**Impact:**
- **Critical**: All chunk files are corrupted during pipeline execution
- **High**: Some chunk files remain corrupted even after completion
- **User-facing**: Files cannot be read by any Parquet reader (DataWrangler, pandas, PyArrow, etc.)

---

### Issue 2: Filename Mismatch (max_file_size logic)

**Location:** `src/datatrove/pipeline/writers/disk_base.py:151-159`

**Problem:**
CheckpointManager closes the "logical" filename, but ParquetWriter writes to a "physical" filename with `000_` prefix added by `_get_filename_with_file_id()` due to `max_file_size` logic.

**Example:**
```python
# DiskWriter.write() calculates:
original_name = "00000.parquet"  # _get_output_filename()
output_filename = "000_00000.parquet"  # _get_filename_with_file_id() when max_file_size > 0

# CheckpointManager tries to close:
filename = "00000.parquet"  # Only has access to _get_output_filename()
output_mg.pop("00000.parquet")  # Wrong file! Data is in "000_00000.parquet"
```

**Why filename differs:**
1. **Step 1**: `_get_output_filename()` - Template substitution only → `"00000.parquet"`
2. **Step 2**: `_get_filename_with_file_id()` - Adds file_id prefix → `"000_00000.parquet"`
3. CheckpointManager only executes Step 1, not Step 2

**Impact:**
- **High**: Wrong file gets closed (or empty file created)
- **Critical**: Actual data file remains open and corrupted
- **Confusing**: Two files with similar names, only one valid

---

### Issue 3: All Chunks Written to Same File

**Root Cause:**
VERL example uses `output_filename="${rank}.parquet"` without `chunk_index`, so all chunks (0, 500, 1000, ...) are appended to the same file.

**Problem:**
- Cannot monitor completed chunks during execution
- File only readable after entire pipeline completes
- Cannot incrementally process results

**User Insight:**
✨ Since CheckpointManager already passes `chunk_index` to `_get_output_filename()` (Line 250), adding `${chunk_index}` to the template would separate chunks into individual files!

**Impact:**
- **Medium**: Poor monitoring experience
- **Low**: Not technically a bug, but missed optimization

---

## 🎯 Reproduction

### Test Case 1: Basic Corruption Test
**File:** `tests/pipeline/test_inference_parquet_corruption.py::test_parquet_corruption_with_chunking`
w
**Setup:**
- 15 documents
- 5 documents per chunk
- ParquetWriter with snappy compression

**Results:**
```
DURING EXECUTION:
  ❌ 00000_chunk_0.parquet - 0 bytes (corrupted)
  ❌ 00000_chunk_1.parquet - 0 bytes (corrupted)
  ❌ 00000_chunk_2.parquet - 0 bytes (corrupted)
  ❌ 000_00000_chunk_0.parquet - 0 bytes (corrupted)
  ❌ 000_00000_chunk_1.parquet - 0 bytes (corrupted)
  ❌ 000_00000_chunk_2.parquet - 0 bytes (corrupted)

AFTER COMPLETION:
  ❌ 00000_chunk_0.parquet - 0 bytes (still corrupted)
  ❌ 00000_chunk_1.parquet - 0 bytes (still corrupted)
  ❌ 00000_chunk_2.parquet - 0 bytes (still corrupted)
  ✓ 000_00000_chunk_0.parquet - 3306 bytes (5 rows, valid)
  ✓ 000_00000_chunk_1.parquet - 3306 bytes (5 rows, valid)
  ✓ 000_00000_chunk_2.parquet - 3317 bytes (5 rows, valid)
```

### Test Case 2: Mid-Execution Corruption Test
**File:** `tests/pipeline/test_parquet_mid_execution.py::test_parquet_corruption_during_execution`

**Setup:**
- Simulates checking files WHILE pipeline is running
- 20 documents, 5 per chunk
- Direct CheckpointManager behavior testing

**Results:**
```
BEFORE writer.close():
  ❌ ALL 8 files corrupted (0 bytes, no magic bytes)

AFTER writer.close():
  ❌ 00000_chunk_X.parquet files still corrupted
  ✓ 000_00000_chunk_X.parquet files recovered
```

**Key Finding:** This matches user's observation that files are unreadable during execution.

---

## 💡 Why This Happens

### ParquetWriter Internal Architecture

```python
class ParquetWriter(DiskWriter):
    def __init__(...):
        self._writers = {}      # PyArrow ParquetWriter instances
        self._batches = {}      # Buffered data not yet written
        self.batch_size = 1000  # Default batch size

    def _write(self, document, file_handler, filename):
        # Add to batch
        self._batches[filename].append(document)

        # Only flush when batch is full
        if len(self._batches[filename]) == self.batch_size:
            self._write_batch(filename)

    def close(self):
        # Flush remaining batches
        for filename in self._batches.keys():
            self._write_batch(filename)
        # Close PyArrow writers (writes footer/metadata)
        for writer in self._writers.values():
            writer.close()
```

### What CheckpointManager Does Wrong

```python
# Only closes file descriptor!
output_writer_context.output_mg.pop(filename).close()

# Does NOT:
# 1. Flush _batches[filename]
# 2. Close _writers[filename]
# 3. Clear ParquetWriter internal state
```

### Parquet File Structure

```
[PAR1 magic bytes - 4 bytes]
[Data pages]
[Column metadata]
[Footer metadata]
[Footer length - 4 bytes]
[PAR1 magic bytes - 4 bytes]
```

**Without proper close:**
- No footer written
- No magic bytes at end
- File appears as 0 bytes or incomplete
- Cannot be read by any Parquet reader

---

## ✅ Solution Requirements

### Must-Have:
1. Flush ParquetWriter's internal `_batches` before closing
2. Properly close ParquetWriter's `_writers` to write footer/metadata
3. Prevent duplicate file creation
4. Maintain backward compatibility with other writers (JsonlWriter, etc.)
5. Preserve checkpoint recovery functionality

### Nice-to-Have:
1. Cleaner abstraction for writer cleanup
2. Better error messages when files are corrupted
3. Validation that files are readable after chunk completion

---

## 🔧 Proposed Solutions

### Solution 1: Add Cleanup Method to DiskWriter (Recommended)

**Approach:**
Add a new `close_file()` method to `DiskWriter` base class that:
1. Handles filename mismatch (max_file_size logic)
2. Allows writers to override for custom cleanup
3. Properly closes ParquetWriter internal state

**Implementation:**

```python
# In src/datatrove/pipeline/writers/disk_base.py

class DiskWriter(PipelineStep, ABC):

    def close_file(self, original_name: str):
        """
        Close a specific file properly.

        - Handles max_file_size logic (calculates actual physical filename)
        - Subclasses override to add custom cleanup logic

        Args:
            original_name: Logical filename from _get_output_filename()
                          (e.g., "00000_chunk_00000.parquet")
        """
        # Calculate actual physical filename (with 000_ prefix if max_file_size > 0)
        if self.max_file_size > 0:
            actual_filename = self._get_filename_with_file_id(original_name)
        else:
            actual_filename = original_name

        # Close file handler
        if actual_filename in self.output_mg:
            self.output_mg.pop(actual_filename).close()
```

```python
# In src/datatrove/pipeline/writers/parquet.py

class ParquetWriter(DiskWriter):

    def close_file(self, original_name: str):
        """Override to properly flush batches and close PyArrow writer."""
        # 1. Flush any remaining batch data
        if original_name in self._batches:
            self._write_batch(original_name)

        # 2. Close PyArrow ParquetWriter (writes footer/metadata)
        if original_name in self._writers:
            self._writers.pop(original_name).close()

        # 3. Close file handler (handles 000_ prefix)
        super().close_file(original_name)
```

```python
# In src/datatrove/pipeline/inference/run_inference.py

class CheckpointManager:

    async def write_document(self, ...):
        if self.per_chunk_counts[chunk_index] == self.records_per_chunk:
            filename = output_writer_context._get_output_filename(
                document, rank, chunk_index=chunk_index
            )

            # Use new cleanup method instead of direct close
            output_writer_context.close_file(filename)

            self.new_completed_chunks.add(chunk_index)
            should_update_last_chunk_index = True
```

**Pros:**
- ✅ Clean abstraction - each writer handles its own cleanup
- ✅ Fixes filename mismatch (max_file_size logic)
- ✅ Properly closes ParquetWriter internal state
- ✅ Backward compatible - default implementation same as current behavior
- ✅ Extends to other writers that might need special cleanup
- ✅ Minimal code changes
- ✅ Easy to understand and maintain

**Cons:**
- Requires changes to base class
- All writer implementations should be reviewed

---

### Solution 2: Add chunk_index to Filename Template (Monitoring Enhancement)

**Approach:**
Separate chunks into individual files by adding `${chunk_index}` to the filename template.

**User Insight:**
✨ CheckpointManager already passes `chunk_index` to `_get_output_filename()`, so this is just a template change!

**Implementation:**

```python
# In examples/verl_data_processing.py

# Before:
output_filename="${rank}.parquet"  # All chunks in one file

# After:
output_filename="${rank}_chunk_${chunk_index}.parquet"  # Each chunk separate
```

**Results:**
```
Before: 000_00000.parquet (all 1500 rows)
After:  000_00000_chunk_00000.parquet (500 rows)
        000_00000_chunk_00001.parquet (500 rows)
        000_00000_chunk_00002.parquet (500 rows)
```

**Pros:**
- ✅ Enables monitoring of completed chunks during execution
- ✅ No code changes, just template change
- ✅ Cleaner logical separation
- ✅ User can choose: chunk files or single file

**Cons:**
- ⚠️ Does NOT fix the corruption bug by itself
- ⚠️ Still needs Solution 1 to work properly
- More files created (not necessarily a problem)

**Important:**
This is **complementary** to Solution 1, not a replacement!
- Solution 1 fixes the corruption bug
- Solution 2 enables better monitoring
- **Both are needed for optimal user experience**

---

## 🎖️ Recommended Solution

**Implement BOTH Solution 1 and Solution 2**

### Why Both Are Needed:

| Issue | Solution 1 (close_file) | Solution 2 (chunk_index) | Status |
|-------|------------------------|-------------------------|---------|
| File Corruption | ✅ Fixes | ❌ Doesn't fix | **Must have** |
| Filename Mismatch | ✅ Fixes | ❌ Doesn't fix | **Must have** |
| Internal State Cleanup | ✅ Fixes | ❌ Doesn't fix | **Must have** |
| Chunk Monitoring | ⚠️ Partial | ✅ Enables | **Nice to have** |
| File Organization | - | ✅ Cleaner | **Nice to have** |

**Conclusion:**
- **Solution 1 is essential** for fixing the bug
- **Solution 2 is optional but recommended** for better UX

### Implementation Steps:

#### Phase 1: Fix Corruption Bug (Required)
1. Add `close_file()` method to `DiskWriter` base class
2. Override in `ParquetWriter` to flush batches and close PyArrow writers
3. Update `CheckpointManager` to use new method
4. Update `_on_file_switch()` for consistency
5. Update tests to verify fix

#### Phase 2: Improve Monitoring (Recommended)
6. Update VERL example to use `"${rank}_chunk_${chunk_index}.parquet"`
7. Add documentation about chunk monitoring options
8. Add comments explaining single-file vs chunk-file trade-offs

### Migration Path:
- **Backward compatible**: Existing code continues to work
- **Incremental adoption**: Users can keep single-file mode or switch to chunk-file mode
- **No breaking changes**: Default behavior preserved

---

## 🧪 Validation Plan

### Unit Tests:
1. ✅ `test_parquet_corruption_with_chunking` - Verify files are valid after chunk completion
2. ✅ `test_parquet_corruption_during_execution` - Verify files are valid mid-execution
3. ✅ `test_parquet_file_split_by_size` - Test file splitting with max_file_size
4. New: `test_diskwriter_close_file_contract` - Verify base class contract

### Integration Tests:
1. Full VERL pipeline with real inference
2. Multiple concurrent tasks
3. Checkpoint recovery scenarios
4. Large datasets (>10GB) with file splitting

### Manual Testing:
1. Run `examples/verl_data_processing.py` with small dataset
2. Check files during execution with DataWrangler/PyArrow
3. Verify checkpoint recovery works
4. Verify final output is complete and valid

---

## 📊 Impact Assessment

### Users Affected:
- Anyone using `InferenceRunner` + `ParquetWriter` + checkpointing
- Particularly VERL data processing workflows
- Any long-running inference jobs that write to Parquet

### Data Loss Risk:
- **Current**: HIGH - Files are corrupted and unreadable
- **After Fix**: NONE - All files will be properly formatted

### Performance Impact:
- **Negligible**: Proper cleanup adds <1ms per chunk
- **Positive**: Prevents duplicate file creation, saves I/O

### Breaking Changes:
- **None**: Backward compatible solution

---

## 📝 Additional Recommendations

### 1. Add File Validation
```python
# Optional: Add validation after chunk completion
def validate_parquet_file(file_path):
    """Verify file has magic bytes and is readable."""
    try:
        import pyarrow.parquet as pq
        with open(file_path, 'rb') as f:
            if f.read(4) != b'PAR1':
                raise ValueError("Missing magic bytes")
        pq.read_table(file_path)
        return True
    except Exception as e:
        logger.error(f"File validation failed: {e}")
        return False
```

### 2. Improve Error Messages
```python
# When users try to read corrupted files
"Parquet file is corrupted (missing magic bytes).
This may happen if the pipeline is still running or was interrupted.
If pipeline completed successfully, this is a bug - please report it."
```

### 3. Documentation Updates
- Add note to InferenceRunner docs about ParquetWriter compatibility
- Document checkpoint behavior with different writers
- Add troubleshooting section for corrupted files

### 4. Consider Atomic Writes
```python
# Write to temporary file, then atomic rename
temp_file = f"{filename}.tmp"
# ... write data ...
os.rename(temp_file, filename)  # Atomic on most filesystems
```

---

## 🚀 Next Steps

1. **Implement Solution 1** (Add cleanup method)
2. **Run all tests** to ensure no regressions
3. **Update documentation**
4. **Create pull request** with detailed explanation
5. **Add to CHANGELOG** as bug fix

---

## 📚 References

- Parquet format spec: https://parquet.apache.org/docs/file-format/
- PyArrow ParquetWriter: https://arrow.apache.org/docs/python/parquet.html
- VERL data format: https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
- Related issues: (to be added if exists)
