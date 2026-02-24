# Hasura Tiling Status Integration

## Overview
The tiling service automatically updates the tiling status in the Hasura database throughout the processing lifecycle.

## Database Schema

### Tables Involved

**`files` table:**
- Main table containing file records
- Primary key: `id`

**`files_information` table:**
- Stores metadata and status information for files
- Columns:
  - `id` (primary key)
  - `files_id` (foreign key to `files.id`)
  - `information_id` (references information type)
  - `value` (text field containing the status)
  - `created_at`, `updated_at` (timestamps)

**Relationship:** `files` → One-To-Many → `files_information`

## Status Values

The tiling service updates the `value` field in `files_information` where `information.name = "tiling-status"` with these statuses:

| Status | When | Description |
|--------|------|-------------|
| `Tiling not started` | Default | Initial status before processing begins |
| `Tiling running` | Start of processing | Background task has started processing the PDF |
| `Tiling finished` | Successful completion | All tiles generated and uploaded successfully |
| `Error` | Failure | Processing failed (includes error message) |

## Configuration

Set these environment variables:

```bash
# Hasura GraphQL endpoint
HASURA_GRAPHQL_URL="https://hasura-blocks-prod.blackpond-228bbd7d.germanywestcentral.azurecontainerapps.io/v1/graphql"

# Hasura admin secret
HASURA_ADMIN_SECRET="admin"
```

## GraphQL Queries Used

### 1. Find Tiling Status Record

```graphql
query FindTilingStatusInfo($files_id: Int!) {
  files_information(where: {
    files_id: {_eq: $files_id}
    information: {name: {_eq: "tiling-status"}}
  }) {
    id
    information_id
    value
  }
}
```

### 2. Update Status

```graphql
mutation UpdateTilingStatus($id: Int!, $value: String!) {
  update_files_information_by_pk(
    pk_columns: {id: $id}
    _set: {
      value: $value
      updated_at: "now()"
    }
  ) {
    id
    value
    updated_at
  }
}
```

## Implementation Details

### Status Update Flow

```
1. API receives /api/process-floorplan request
   │
   ├─ Validates URL and file_id
   │
2. Job queued → process_floorplan_background() starts
   │
   ├─ Status: "Tiling running"
   │  └─ Hasura update: files_information.value = "Tiling running"
   │
3. process_floorplan_sync() executes
   │
   ├─ Downloads PDF
   ├─ Converts to images
   ├─ Generates tiles
   ├─ Uploads to Azure Storage
   │
4. Success or Failure
   │
   ├─ Success:
   │  └─ Hasura update: files_information.value = "Tiling finished"
   │
   └─ Failure:
      └─ Hasura update: files_information.value = "Error: <error message>"
```

### Function: `update_tiling_status()`

Located in `app.py`, this async function:

1. **Finds the record**: Queries `files_information` to find the record where:
   - `files_id` matches the provided `file_id`
   - `information.name = "tiling-status"`

2. **Updates the value**: Executes mutation to update:
   - `value` field with the new status
   - `updated_at` timestamp

3. **Error handling**: 
   - Logs warnings if Hasura config is missing
   - Logs errors but doesn't fail the tiling process
   - Timeouts are handled gracefully (10s timeout)

### Error Message Format

When status is "Error", the value includes the error message:
```
Error: PDF file too large (150.5MB). Maximum allowed size is 100MB.
```

Error messages are truncated to 200 characters to fit database constraints.

## Code Integration Points

### app.py Changes

**1. Configuration (Lines 36-38):**
```python
HASURA_GRAPHQL_URL = os.environ.get("HASURA_GRAPHQL_URL", "...")
HASURA_ADMIN_SECRET = os.environ.get("HASURA_ADMIN_SECRET", "admin")
```

**2. Status Update Function (Lines 67-160):**
```python
async def update_tiling_status(file_id: int, status: str, error_message: Optional[str] = None):
    # GraphQL queries and mutations
    # Error handling
    # Logging
```

**3. Background Processing (Lines 890-920):**
```python
def process_floorplan_background(job_id: str, file_url: str, file_id: int, environment: str):
    try:
        # Status: "Tiling running"
        asyncio.run(update_tiling_status(file_id, "Tiling running"))
        
        result = process_floorplan_sync(...)
        
        # Status: "Tiling finished"
        asyncio.run(update_tiling_status(file_id, "Tiling finished"))
        
    except Exception as e:
        # Status: "Error"
        asyncio.run(update_tiling_status(file_id, "Error", error_message=str(e)))
```

## Testing

### Manual Test via API

```bash
curl -X POST http://localhost:8000/api/process-floorplan \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "file_url": "https://example.com/floorplan.pdf",
    "file_id": 123,
    "environment": "test"
  }'
```

### Verify in Hasura Console

1. Open Hasura Console: `https://hasura-blocks-prod.blackpond-228bbd7d.germanywestcentral.azurecontainerapps.io/console`

2. Run query:
```graphql
query CheckTilingStatus($files_id: Int!) {
  files_information(where: {
    files_id: {_eq: $files_id}
    information: {name: {_eq: "tiling-status"}}
  }) {
    id
    files_id
    value
    updated_at
  }
}
```

3. Variables:
```json
{
  "files_id": 123
}
```

## Logging

Status updates are logged with these patterns:

**Success:**
```
✅ Updated tiling status for file_id=123: 'Tiling running'
✅ Updated tiling status for file_id=123: 'Tiling finished'
```

**Warnings:**
```
Hasura configuration not set, skipping status update
No tiling-status record found for file_id=123
```

**Errors:**
```
Hasura query error: [error details]
Hasura mutation error: [error details]
Timeout updating tiling status for file_id=123
HTTP error updating tiling status for file_id=123: [details]
```

## Failure Modes

### Graceful Degradation

The tiling service continues to work even if Hasura updates fail:

1. **Missing Config**: Logs warning, continues processing
2. **Network Timeout**: Logs error, continues processing
3. **Invalid Response**: Logs error, continues processing
4. **Missing Record**: Logs warning, continues processing

**This ensures tiling jobs complete successfully even if status tracking fails.**

## Security

- Uses `x-hasura-admin-secret` header for authentication
- Admin secret should be stored in environment variables
- Never hardcode credentials in source code
- Use Azure Key Vault in production:

```bash
# Production deployment
az containerapp secret set \
  --name floorplan-tiler \
  --resource-group rg-floorplan-tiler \
  --secrets hasura-admin-secret=<value-from-key-vault>
```

## Troubleshooting

### Status not updating?

1. Check Hasura configuration:
```python
print(f"HASURA_GRAPHQL_URL: {os.environ.get('HASURA_GRAPHQL_URL')}")
print(f"HASURA_ADMIN_SECRET: {'***' if os.environ.get('HASURA_ADMIN_SECRET') else 'NOT SET'}")
```

2. Verify `tiling-status` record exists in database

3. Check logs for Hasura errors

4. Test Hasura endpoint manually:
```bash
curl -X POST https://hasura-blocks-prod.../v1/graphql \
  -H "Content-Type: application/json" \
  -H "x-hasura-admin-secret: admin" \
  -d '{"query":"{ files_information { id files_id value } }"}'
```

### Wrong status showing?

- Check `updated_at` timestamp to see when it was last modified
- Verify `file_id` matches between API call and database record
- Check for duplicate records with same `files_id`

## Future Enhancements

Potential improvements:

1. **Progress tracking**: Update status with percentage (e.g., "Tiling running: 45%")
2. **Detailed stages**: Track individual stages ("Downloading", "Converting", "Uploading tiles")
3. **Retry logic**: Automatic retry for failed Hasura updates
4. **Batch updates**: Queue multiple status updates and send in batch
5. **Webhooks**: Trigger webhooks when status changes
