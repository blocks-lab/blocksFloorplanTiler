// Main Bicep template for Blocks Floorplan Tiler Service
// Deploys Azure Container Apps with all required resources

targetScope = 'resourceGroup'

@description('Name of the application')
param appName string = 'blocks-floorplan-tiler'

@description('Location for all resources')
param location string = resourceGroup().location

@description('Environment name (dev, staging, prod)')
@allowed([
  'dev'
  'staging'
  'prod'
])
param environmentName string = 'dev'

@description('Container image to deploy (e.g., your-acr.azurecr.io/floorplan-tiler:latest)')
param containerImage string

@description('Azure Storage connection string for floor-plans container')
@secure()
param storageConnectionString string

@description('Hasura GraphQL endpoint URL')
param hasuraGraphqlUrl string = 'https://hasura-blocks-prod.blackpond-228bbd7d.germanywestcentral.azurecontainerapps.io/v1/graphql'

@description('Hasura admin secret for authentication')
@secure()
param hasuraAdminSecret string

@description('API key for securing the tiling service endpoints')
@secure()
param apiKey string = ''

@description('Minimum number of replicas')
@minValue(0)
@maxValue(30)
param minReplicas int = 0

@description('Maximum number of replicas')
@minValue(1)
@maxValue(1000)
param maxReplicas int = 50

@description('CPU cores per container instance')
@allowed([
  '0.25'
  '0.5'
  '0.75'
  '1'
  '1.25'
  '1.5'
  '1.75'
  '2'
  '2.5'
  '3'
  '3.5'
  '4'
])
param cpuCores string = '2'

@description('Memory in GB per container instance (must maintain 2:1 ratio with CPU)')
@allowed([
  '0.5'
  '1'
  '1.5'
  '2'
  '3'
  '4'
  '6'
  '8'
])
param memorySize string = '4'

// Variables
var uniqueSuffix = uniqueString(resourceGroup().id)
var logAnalyticsName = '${appName}-logs-${uniqueSuffix}'
var containerAppEnvName = '${appName}-env-${environmentName}'
var containerAppName = '${appName}-${environmentName}'

// Log Analytics Workspace for Container Apps logging
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Container Apps Environment
resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// Container App - Floorplan Tiler Service
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: containerAppName
  location: location
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      secrets: [
        {
          name: 'storage-connection-string'
          value: storageConnectionString
        }
        {
          name: 'hasura-admin-secret'
          value: hasuraAdminSecret
        }
        {
          name: 'api-key'
          value: apiKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'floorplan-tiler'
          image: containerImage
          resources: {
            cpu: json(cpuCores)
            memory: '${memorySize}Gi'
          }
          env: [
            {
              name: 'AZURE_STORAGE_CONNECTION_STRING'
              secretRef: 'storage-connection-string'
            }
            {
              name: 'HASURA_GRAPHQL_URL'
              value: hasuraGraphqlUrl
            }
            {
              name: 'HASURA_ADMIN_SECRET'
              secretRef: 'hasura-admin-secret'
            }
            {
              name: 'API_KEY'
              secretRef: 'api-key'
            }
            {
              name: 'PORT'
              value: '8000'
            }
            {
              name: 'PYTHONUNBUFFERED'
              value: '1'
            }
          ]
          probes: [
            {
              type: 'liveness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 30
              periodSeconds: 30
              timeoutSeconds: 10
              failureThreshold: 3
            }
            {
              type: 'readiness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        cooldownPeriodInSeconds: 300
        pollingIntervalInSeconds: 30
        rules: [
          {
            name: 'http-rule'
            http: {
              metadata: {
                concurrentRequests: '1'
              }
            }
          }
        ]
      }
    }
  }
}

// Outputs
output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output containerAppFqdn string = containerApp.properties.configuration.ingress.fqdn
output containerAppName string = containerApp.name
output containerAppEnvName string = containerAppEnv.name
output logAnalyticsWorkspaceId string = logAnalytics.id
