"""
Worker for processing floorplan tiling queue messages.

Runs as a separate Container App (same image as API, different command: python worker.py).
Polls Azure Storage Queue, processes one tiling job at a time, and exits when the queue
is empty. KEDA's azure-queue scaler on this Container App handles lifecycle:
  - queue depth > 0  →  scale from 0 to 1 replica  (worker starts)
  - queue depth = 0  →  scale from 1 to 0 replicas (worker exits, cost = $0)

Sequential processing (maxReplicas=1) means only one PDF is tiled at a time,
so memory usage is predictable and OOM crashes are impossible.

Required environment variables:
  AZURE_STORAGE_CONNECTION_STRING       - queue storage + test blob storage
                                          ("tileservice" queue must exist here)
  PRODUCTION_STORAGE_CONNECTION_STRING  - fallback if AZURE_STORAGE_CONNECTION_STRING is absent;
                                          also used for production blob uploads
  TILING_QUEUE_NAME                     - queue name (default: tileservice)
  PRODUCTION_STORAGE_ACCOUNT_NAME       - production account name
  TEST_STORAGE_ACCOUNT_NAME             - test account name
  HASURA_GRAPHQL_URL                    - for tiling status updates
  HASURA_ADMIN_SECRET                   - for tiling status updates
"""
import json
import logging
import os
import sys
import time
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prefer AZURE_STORAGE_CONNECTION_STRING (same account the API enqueues to),
# fall back to PRODUCTION_STORAGE_CONNECTION_STRING for backwards compatibility.
QUEUE_STORAGE_CONNECTION_STRING = (
    os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    or os.environ.get("PRODUCTION_STORAGE_CONNECTION_STRING")
)
TILING_QUEUE_NAME = os.environ.get("TILING_QUEUE_NAME", "tileservice")

# Shut down after this many consecutive empty polls (KEDA will restart when needed)
EMPTY_POLL_LIMIT = 3
EMPTY_POLL_WAIT_SECONDS = 10


def main():
    if not QUEUE_STORAGE_CONNECTION_STRING:
        logger.error("❌ AZURE_STORAGE_CONNECTION_STRING (or PRODUCTION_STORAGE_CONNECTION_STRING) not set")
        sys.exit(1)

    # Lazy import: avoids loading heavy deps (PIL, PyMuPDF) if env is misconfigured
    from azure.storage.queue import QueueServiceClient

    from app import process_floorplan_sync, update_tiling_status

    queue_service = QueueServiceClient.from_connection_string(QUEUE_STORAGE_CONNECTION_STRING)
    queue_client = queue_service.get_queue_client(TILING_QUEUE_NAME)

    logger.info(f"🚀 Worker started | queue={TILING_QUEUE_NAME}")

    consecutive_empty = 0

    while True:
        # visibility_timeout=1800: message reappears after 30 min if worker crashes
        message = next(
            iter(queue_client.receive_messages(max_messages=1, visibility_timeout=1800)),
            None
        )

        if message is None:
            consecutive_empty += 1
            if consecutive_empty >= EMPTY_POLL_LIMIT:
                logger.info(
                    f"✅ Queue empty after {EMPTY_POLL_LIMIT} consecutive polls - "
                    f"worker shutting down (KEDA will restart when new messages arrive)"
                )
                break
            logger.info(
                f"📭 Queue empty (poll {consecutive_empty}/{EMPTY_POLL_LIMIT}), "
                f"waiting {EMPTY_POLL_WAIT_SECONDS}s..."
            )
            time.sleep(EMPTY_POLL_WAIT_SECONDS)
            continue

        consecutive_empty = 0
        payload = None

        try:
            payload = json.loads(message.content)
            file_url = payload["file_url"]
            file_id = int(payload["file_id"])
            environment = payload.get("environment", "test")
            job_id = payload.get("job_id", str(uuid.uuid4()))

            logger.info(
                f"📋 Processing | file_id={file_id} | environment={environment} | job_id={job_id}"
            )

            update_tiling_status(file_id, "Tiling running")
            process_floorplan_sync(file_url, job_id, file_id, environment)
            update_tiling_status(file_id, "Tiling finished")

            queue_client.delete_message(message)
            logger.info(f"✅ Completed | file_id={file_id}")

        except Exception as e:
            file_id_log = str(payload.get("file_id", "unknown")) if payload else "unknown"
            logger.error(f"❌ Failed | file_id={file_id_log} | error={e}", exc_info=True)

            # Update Hasura status to Error
            try:
                if payload and payload.get("file_id"):
                    update_tiling_status(int(payload["file_id"]), "Error", error_message=str(e))
            except Exception:
                pass

            # Always delete the message to avoid a poison-pill infinite loop
            try:
                queue_client.delete_message(message)
            except Exception:
                pass


if __name__ == "__main__":
    main()
