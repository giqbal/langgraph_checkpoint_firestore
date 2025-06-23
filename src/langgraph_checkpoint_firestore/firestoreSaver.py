# create firestore Saver (langgraph checkpointer) basis firestoredb implementation in included code include=src/langgraph_checkpoint_firestore/firestoredbSaver.py
#  Remember there is no partition key in firestore only unique ID.
# create separate collection for checkpoints and writes. Next level collection will be thread_id and then checkpoint_id
# we have to preserve the function signatures and return values (as in firestoredbsaver) as it is.
# Do provide full and complete code, i.e all function along with Saver in the included file.
# @!

import asyncio
import random
import time
from contextlib import contextmanager
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple

from google.api_core import exceptions
from google.cloud import firestore
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
)

from langgraph_checkpoint_firestore.firestoreSerializer import FirestoreSerializer

FIRESTORE_KEY_SEPARATOR = "/"

# Retry configuration for Firestore operations
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 10.0  # seconds


def exponential_backoff_with_jitter(
    attempt: int, base_delay: float = BASE_DELAY, max_delay: float = MAX_DELAY
) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = min(base_delay * (2**attempt), max_delay)
    jitter = random.uniform(0, 0.1 * delay)
    return delay + jitter


def should_retry_exception(exception: Exception) -> bool:
    """Determine if an exception should trigger a retry."""
    if isinstance(exception, exceptions.InternalServerError):
        return True
    if isinstance(exception, exceptions.ServiceUnavailable):
        return True
    if isinstance(exception, exceptions.DeadlineExceeded):
        return True
    if isinstance(exception, exceptions.Aborted):
        return True
    if isinstance(exception, exceptions.ResourceExhausted):
        return True
    return False


def retry_firestore_operation(operation, *args, **kwargs):
    """Retry a Firestore operation with exponential backoff."""
    last_exception = None

    for attempt in range(MAX_RETRIES):
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if not should_retry_exception(e):
                raise e

            if attempt < MAX_RETRIES - 1:
                delay = exponential_backoff_with_jitter(attempt)
                print(
                    f"Firestore operation failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                )
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print(f"Firestore operation failed after {MAX_RETRIES} attempts: {e}")
                raise last_exception

    raise last_exception


def _make_firestore_checkpoint_key(
    thread_id: str, checkpoint_ns: str, checkpoint_id: str
) -> str:
    return FIRESTORE_KEY_SEPARATOR.join(
        ["checkpoint", thread_id, checkpoint_ns, checkpoint_id]
    )


def _make_firestore_checkpoint_writes_key(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: Optional[int],
) -> str:
    if idx is None:
        return FIRESTORE_KEY_SEPARATOR.join(
            ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id]
        )

    return FIRESTORE_KEY_SEPARATOR.join(
        ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id, str(idx)]
    )


def _parse_firestore_checkpoint_key(firestoredb_key: str) -> dict:
    namespace, thread_id, checkpoint_ns, checkpoint_id = firestoredb_key.split(
        FIRESTORE_KEY_SEPARATOR
    )
    if namespace != "checkpoint":
        raise ValueError("Expected checkpoint key to start with 'checkpoint'")

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
    }


def _parse_firestore_checkpoint_writes_key(firestoredb_key: str) -> dict:
    namespace, thread_id, checkpoint_ns, checkpoint_id, task_id, idx = (
        firestoredb_key.split(FIRESTORE_KEY_SEPARATOR)
    )
    if namespace != "writes":
        raise ValueError("Expected checkpoint key to start with 'writes'")

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
        "task_id": task_id,
        "idx": idx,
    }


def _filter_keys(
    keys: List[str], before: Optional[RunnableConfig], limit: Optional[int]
) -> list:
    if before:
        keys = [
            k
            for k in keys
            if _parse_firestore_checkpoint_key(k)["checkpoint_id"]
            < before["configurable"]["checkpoint_id"]
        ]

    keys = sorted(
        keys,
        key=lambda k: _parse_firestore_checkpoint_key(k)["checkpoint_id"],
        reverse=True,
    )
    if limit:
        keys = keys[:limit]
    return keys


def _load_writes(
    serde: FirestoreSerializer, task_id_to_data: dict[tuple[str, str], dict]
) -> list[PendingWrite]:
    writes = [
        (
            task_id,
            data["channel"],
            serde.loads_typed((data["type"], data["value"])),
        )
        for (task_id, _), data in task_id_to_data.items()
    ]
    return writes


def _parse_firestore_checkpoint_data(
    serde: FirestoreSerializer,
    key: str,
    data: dict,
    pending_writes: Optional[List[PendingWrite]] = None,
) -> Optional[CheckpointTuple]:
    if not data:
        return None

    parsed_key = _parse_firestore_checkpoint_key(key)
    thread_id = parsed_key["thread_id"]
    checkpoint_ns = parsed_key["checkpoint_ns"]
    checkpoint_id = parsed_key["checkpoint_id"]
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }

    checkpoint = serde.loads_typed((data["type"], data["checkpoint"]))

    metadata = serde.loads(data["metadata"])
    parent_checkpoint_id = data.get("parent_checkpoint_id", "")
    parent_config = (
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
        if parent_checkpoint_id
        else None
    )
    return CheckpointTuple(
        config=config,
        checkpoint=checkpoint,
        metadata=metadata,
        parent_config=parent_config,
        pending_writes=pending_writes,
    )


class FirestoreSaver(BaseCheckpointSaver):
    def __init__(
        self,
        project_id,
        checkpoints_collection="checkpoints",
        writes_collection="writes",
        max_retries=3,
        base_delay=1.0,
        max_delay=10.0,
        enable_retries=True,
    ):
        super().__init__()
        self.client = firestore.Client(project=project_id)
        self.firestore_serde = FirestoreSerializer(self.serde)
        self.checkpoints_collection = self.client.collection(checkpoints_collection)
        self.writes_collection = self.client.collection(writes_collection)

        # Retry configuration
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.enable_retries = enable_retries

    @classmethod
    @contextmanager
    def from_conn_info(
        cls, *, project_id: str, checkpoints_collection: str, writes_collection: str
    ) -> Iterator["FirestoreSaver"]:
        saver = None
        try:
            saver = FirestoreSaver(
                project_id, checkpoints_collection, writes_collection
            )
            yield saver
        finally:
            pass

    def _retry_operation(self, operation, *args, **kwargs):
        """Retry operation with configured settings."""
        if not self.enable_retries:
            return operation(*args, **kwargs)

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if not should_retry_exception(e):
                    raise e

                if attempt < self.max_retries - 1:
                    delay = exponential_backoff_with_jitter(
                        attempt, self.base_delay, self.max_delay
                    )
                    print(
                        f"Firestore operation failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    print(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print(
                        f"Firestore operation failed after {self.max_retries} attempts: {e}"
                    )
                    raise last_exception

        raise last_exception

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "default")

        # FIX: Handle empty string checkpoint_ns - this is CRITICAL for Firestore
        if not checkpoint_ns or checkpoint_ns == "":
            checkpoint_ns = "default"

        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        key = _make_firestore_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        type_, serialized_checkpoint = self.firestore_serde.dumps_typed(checkpoint)
        serialized_metadata = self.firestore_serde.dumps(metadata)
        data = {
            "checkpoint": serialized_checkpoint,
            "checkpoint_key": key,
            "type": type_,
            "metadata": serialized_metadata,
            "parent_checkpoint_id": parent_checkpoint_id
            if parent_checkpoint_id
            else "",
        }

        # Use retry logic for Firestore operation
        def firestore_set_operation():
            return (
                self.checkpoints_collection.document(thread_id)
                .collection(checkpoint_ns)
                .document(checkpoint_id)
                .set(data)
            )

        self._retry_operation(firestore_set_operation)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self, config: RunnableConfig, writes: List[Tuple[str, Any]], task_id: str
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "default")

        # FIX: Handle empty string checkpoint_ns - this is CRITICAL for Firestore
        if not checkpoint_ns or checkpoint_ns == "":
            checkpoint_ns = "default"

        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            key = _make_firestore_checkpoint_writes_key(
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
            )
            type_, serialized_value = self.firestore_serde.dumps_typed(value)
            data = {
                "checkpoint_key": key,
                "channel": channel,
                "type": type_,
                "value": serialized_value,
            }

            # Use retry logic for Firestore operation
            def firestore_write_operation():
                return (
                    self.writes_collection.document(thread_id)
                    .collection(checkpoint_ns)
                    .document(checkpoint_id)
                    .collection(task_id)
                    .document(str(idx))
                    .set(data)
                )

            self._retry_operation(firestore_write_operation)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)

        checkpoint_ns = config["configurable"].get("checkpoint_ns", "default")

        # FIX: Handle empty string checkpoint_ns - this is CRITICAL for Firestore
        if not checkpoint_ns or checkpoint_ns == "":
            checkpoint_ns = "default"

        checkpoint_key = self._get_checkpoint_key(
            thread_id, checkpoint_ns, checkpoint_id
        )

        if not checkpoint_key:
            return None
        checkpoint_id = _parse_firestore_checkpoint_key(checkpoint_key)["checkpoint_id"]

        doc_ref = (
            self.checkpoints_collection.document(thread_id)
            .collection(checkpoint_ns)
            .document(checkpoint_id)
        )

        # Use retry logic for Firestore read operation
        def firestore_get_operation():
            return doc_ref.get()

        doc = self._retry_operation(firestore_get_operation)

        if not doc.exists:
            return None

        checkpoint_data = doc.to_dict()

        pending_writes = self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )
        return _parse_firestore_checkpoint_data(
            self.firestore_serde,
            checkpoint_key,
            checkpoint_data,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "default")

        # FIX: Handle empty string checkpoint_ns - this is CRITICAL for Firestore
        if not checkpoint_ns or checkpoint_ns == "":
            checkpoint_ns = "default"

        # FIX: Use namespace filtering - this is CRITICAL for subgraph state listing
        checkpoints_ref = self.checkpoints_collection.document(thread_id).collection(
            checkpoint_ns
        )

        # Use retry logic for Firestore read operation
        def firestore_stream_operation():
            return checkpoints_ref.stream()

        checkpoints = self._retry_operation(firestore_stream_operation)

        for checkpoint_doc in checkpoints:
            checkpoint_id = checkpoint_doc.id
            checkpoint_data = checkpoint_doc.to_dict()
            if checkpoint_data:
                pending_writes = self._load_pending_writes(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                yield _parse_firestore_checkpoint_data(
                    self.firestore_serde,
                    checkpoint_data["checkpoint_key"],
                    checkpoint_data,
                    pending_writes,
                )

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: Optional[str], checkpoint_id: str
    ) -> List[PendingWrite]:
        # FIX: Ensure checkpoint_ns is never empty or None
        if not checkpoint_ns:
            checkpoint_ns = "default"

        # FIX: Use namespace filtering - this is CRITICAL for subgraph writes retrieval
        writes_ref = (
            self.writes_collection.document(thread_id)
            .collection(checkpoint_ns)
            .document(checkpoint_id)
        )

        # Use retry logic for Firestore read operation
        def firestore_collections_operation():
            return writes_ref.collections()

        writes_docs = self._retry_operation(firestore_collections_operation)

        matching_keys = []
        for task_collection in writes_docs:
            task_id = task_collection.id

            # Use retry logic for Firestore stream operation
            def firestore_task_stream_operation():
                return task_collection.stream()

            task_stream = self._retry_operation(firestore_task_stream_operation)

            for write_doc in task_stream:
                write_data = write_doc.to_dict()
                if write_data:
                    matching_keys.append(write_data)

        parsed_keys = [
            _parse_firestore_checkpoint_writes_key(key["checkpoint_key"])
            for key in matching_keys
        ]
        pending_writes = _load_writes(
            self.firestore_serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): key
                for key, parsed_key in sorted(
                    zip(matching_keys, parsed_keys), key=lambda x: x[1]["idx"]
                )
            },
        )
        return pending_writes

    def _get_checkpoint_key(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: Optional[str]
    ) -> Optional[str]:
        if not checkpoint_ns:
            checkpoint_ns = "default"
        if checkpoint_id:
            return _make_firestore_checkpoint_key(
                thread_id, checkpoint_ns, checkpoint_id
            )

        # FIX: Use namespace filtering - this is CRITICAL for subgraph checkpoint key retrieval
        collections_ref = self.checkpoints_collection.document(thread_id).collection(
            checkpoint_ns
        )

        # Use retry logic for Firestore read operation
        def firestore_stream_operation():
            return collections_ref.stream()

        collections = self._retry_operation(firestore_stream_operation)

        docs = []

        for doc in collections:
            doc_data = doc.to_dict()
            if doc_data and "checkpoint_key" in doc_data:
                docs.append(doc_data)

        if not docs:
            return None

        latest_doc = max(
            docs,
            key=lambda doc: _parse_firestore_checkpoint_key(doc["checkpoint_key"])[
                "checkpoint_id"
            ],
        )
        return latest_doc["checkpoint_key"]

    async def aget(self, config: RunnableConfig) -> Optional[Checkpoint]:
        if value := await self.aget_tuple(config):
            return value.checkpoint

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.get_tuple, config
        )

    async def alist(self, config: RunnableConfig) -> AsyncIterator[CheckpointTuple]:
        loop = asyncio.get_running_loop()
        iter = loop.run_in_executor(None, self.list, config)
        while True:
            try:
                yield await loop.run_in_executor(None, next, iter)
            except StopIteration:
                return

    async def aput(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> RunnableConfig:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint
        )
