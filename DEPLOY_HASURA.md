# Deploy with Hasura Integration to Azure Container Apps

## Quick Deploy Steps

### 1. Update Parameters File

Edit `infra/main.parameters.json`:

```json
{
  "hasuraAdminSecret": {
    "value": "admin"  // Change in production!
  },
  "apiKey": {
    "value": "your-secure-api-key-here"  // Optional but recommended
  }
}
```

### 2. Deploy via Azure CLI

```bash
# Set your resource group and ACR name
RESOURCE_GROUP="rg-floorplan-tiler"
ACR_NAME="yourregistry"

# Build and push image to ACR
az acr build \
  --registry $ACR_NAME \
  --image floorplan-tiler:latest \
  --file Dockerfile \
  .

# Deploy infrastructure with Hasura config
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file infra/main.bicep \
  --parameters infra/main.parameters.json
```

### 3. Verify Deployment

```bash
# Get the app URL
az containerapp show \
  --name blocks-floorplan-tiler-prod \
  --resource-group $RESOURCE_GROUP \
  --query "properties.configuration.ingress.fqdn" \
  -o tsv

# Test health endpoint
curl https://YOUR-APP-URL/health
```

## Environment Variables Configured

The Bicep template automatically sets these environment variables in your Container App:

| Variable | Source | Description |
|----------|--------|-------------|
| `HASURA_GRAPHQL_URL` | Parameter (default set) | Hasura endpoint URL |
| `HASURA_ADMIN_SECRET` | Secret (from parameters) | Hasura admin password |
| `API_KEY` | Secret (from parameters) | Tiling service API key |
| `AZURE_STORAGE_CONNECTION_STRING` | Secret (from parameters) | Azure Storage connection |

## Update Existing Deployment

If you already have a deployed Container App, update it with:

```bash
az containerapp update \
  --name blocks-floorplan-tiler-prod \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars \
    "HASURA_GRAPHQL_URL=https://hasura-blocks-prod.blackpond-228bbd7d.germanywestcentral.azurecontainerapps.io/v1/graphql"

# Add secret for Hasura admin password
az containerapp secret set \
  --name blocks-floorplan-tiler-prod \
  --resource-group $RESOURCE_GROUP \
  --secrets hasura-admin-secret=admin

# Set the environment variable to use the secret
az containerapp update \
  --name blocks-floorplan-tiler-prod \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars "HASURA_ADMIN_SECRET=secretref:hasura-admin-secret"
```

## Verify Hasura Integration

### Test the Integration

```bash
# Start a tiling job
curl -X POST https://YOUR-APP-URL/api/process-floorplan \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "file_url": "https://example.com/test.pdf",
    "file_id": 123,
    "environment": "test"
  }'

# Response will include job_id
{
  "job_id": "abc-123-def",
  "status": "queued",
  "status_url": "/api/status/abc-123-def"
}
```

### Check Status in Hasura

```graphql
query CheckStatus {
  files_information(where: {
    files_id: {_eq: 123}
    information: {name: {_eq: "tiling-status"}}
  }) {
    id
    value
    updated_at
  }
}
```

Should show:
- `"Tiling running"` → while processing
- `"Tiling finished"` → when complete
- `"Error: ..."` → if failed

## View Logs

```bash
# Stream live logs
az containerapp logs tail \
  --name blocks-floorplan-tiler-prod \
  --resource-group $RESOURCE_GROUP \
  --follow

# Look for these log messages:
# ✅ Updated tiling status for file_id=123: 'Tiling running'
# ✅ Updated tiling status for file_id=123: 'Tiling finished'
```

## Production Checklist

- [ ] Change `hasuraAdminSecret` from "admin" to a strong password
- [ ] Set a secure `apiKey` for the tiling service
- [ ] Use Azure Key Vault for secrets (recommended):
  ```bash
  az containerapp secret set \
    --name blocks-floorplan-tiler-prod \
    --resource-group $RESOURCE_GROUP \
    --secrets hasura-admin-secret=keyvaultref:YOUR_KEY_VAULT_URL,identityref:YOUR_IDENTITY
  ```
- [ ] Test with a real PDF file
- [ ] Verify Hasura updates appear in database
- [ ] Check Container App logs for errors

## Troubleshooting

### Status not updating?

1. **Check environment variables:**
   ```bash
   az containerapp show \
     --name blocks-floorplan-tiler-prod \
     --resource-group $RESOURCE_GROUP \
     --query "properties.template.containers[0].env" \
     -o table
   ```

2. **Check logs for Hasura errors:**
   ```bash
   az containerapp logs tail \
     --name blocks-floorplan-tiler-prod \
     --resource-group $RESOURCE_GROUP \
     --follow | grep -i hasura
   ```

3. **Verify Hasura endpoint is accessible from Container App:**
   ```bash
   # Execute command inside container
   az containerapp exec \
     --name blocks-floorplan-tiler-prod \
     --resource-group $RESOURCE_GROUP \
     --command "curl -v https://hasura-blocks-prod.blackpond-228bbd7d.germanywestcentral.azurecontainerapps.io/v1/graphql"
   ```

### Common Issues

**"Hasura configuration not set"**
- Environment variables not configured
- Solution: Check deployment parameters

**"No tiling-status record found"**
- Database doesn't have the information record
- Solution: Ensure `files_information` has a record with `information.name = "tiling-status"` for the file_id

**"Timeout updating tiling status"**
- Network connectivity issue
- Solution: Check Container App can reach Hasura endpoint

## Rollback

If something goes wrong, rollback to previous revision:

```bash
# List revisions
az containerapp revision list \
  --name blocks-floorplan-tiler-prod \
  --resource-group $RESOURCE_GROUP \
  -o table

# Activate previous revision
az containerapp revision activate \
  --revision blocks-floorplan-tiler-prod--PREVIOUS_REVISION \
  --resource-group $RESOURCE_GROUP
```
