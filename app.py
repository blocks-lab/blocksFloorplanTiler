"""
FastAPI application for floorplan PDF tiling service.
Converted from Azure Functions for deployment to Azure Container Apps.
"""
import io
import json
import logging
import math
import os
import threading
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import httpx
import uvicorn
from azure.storage.blob import BlobServiceClient, ContentSettings
from fastapi import (BackgroundTasks, Depends, FastAPI, Header, HTTPException,
                     Request)
from fastapi.responses import JSONResponse
from PIL import Image, ImageChops
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable PIL decompression bomb check for large floor plans
Image.MAX_IMAGE_PIXELS = None

# API Key configuration
API_KEY = os.environ.get("API_KEY", "")  # Set via environment variable

# Hasura configuration for status updates
HASURA_GRAPHQL_URL = os.environ.get("HASURA_GRAPHQL_URL", "https://hasura-blocks-prod.blackpond-228bbd7d.germanywestcentral.azurecontainerapps.io/v1/graphql")
HASURA_ADMIN_SECRET = os.environ.get("HASURA_ADMIN_SECRET", "admin")

# Storage configuration
TEST_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")  # Test/dev storage
TEST_STORAGE_ACCOUNT_NAME = os.environ.get("TEST_STORAGE_ACCOUNT_NAME", "blocksplayground")  # Default test account

PRODUCTION_STORAGE_CONNECTION_STRING = os.environ.get("PRODUCTION_STORAGE_CONNECTION_STRING")  # Production storage
PRODUCTION_STORAGE_ACCOUNT_NAME = os.environ.get("PRODUCTION_STORAGE_ACCOUNT_NAME")  # Production account name

def verify_api_key(x_api_key: str = Header(None)):
    """Verify the API key from request header"""
    if not API_KEY:
        # If no API key is configured, allow all requests (backward compatibility)
        return True

    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key is required. Please provide X-API-Key header."
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return True


def update_tiling_status(file_id: int, status: str, error_message: Optional[str] = None):
    """
    Update tiling status in Hasura database via GraphQL mutation.
    Simple synchronous function - just 3 calls: start, end, error
    
    Args:
        file_id: The file ID to update
        status: One of "Tiling running", "Tiling finished", "Error"
        error_message: Optional error message if status is "Error"
    """
    if not HASURA_GRAPHQL_URL or not HASURA_ADMIN_SECRET:
        logger.warning("Hasura configuration not set, skipping status update")
        return
    
    try:
        # Find the tiling-status record
        find_query = """
        query FindTilingStatusInfo($files_id: Int!) {
          files_information(where: {
            files_id: {_eq: $files_id}
            information: {name: {_eq: "tiling-status"}}
          }) {
            id
          }
        }
        """
        
        find_response = httpx.post(
            HASURA_GRAPHQL_URL,
            headers={
                "Content-Type": "application/json",
                "x-hasura-admin-secret": HASURA_ADMIN_SECRET
            },
            json={
                "query": find_query,
                "variables": {"files_id": file_id}
            },
            timeout=10.0
        )
        find_response.raise_for_status()
        find_data = find_response.json()
        
        if "errors" in find_data:
            logger.error(f"Hasura query error: {find_data['errors']}")
            return
        
        records = find_data.get("data", {}).get("files_information", [])
        if not records:
            logger.warning(f"No tiling-status record found for file_id={file_id}")
            return
        
        record_id = records[0]["id"]
        
        # Build status value
        status_value = status
        if error_message and status == "Error":
            status_value = f"{status}: {error_message[:200]}"
        
        # Update the status
        update_mutation = """
        mutation UpdateTilingStatus($id: Int!, $value: String!) {
          update_files_information_by_pk(
            pk_columns: {id: $id}
            _set: {value: $value}
          ) {
            id
            value
          }
        }
        """
        
        update_response = httpx.post(
            HASURA_GRAPHQL_URL,
            headers={
                "Content-Type": "application/json",
                "x-hasura-admin-secret": HASURA_ADMIN_SECRET
            },
            json={
                "query": update_mutation,
                "variables": {"id": record_id, "value": status_value}
            },
            timeout=10.0
        )
        update_response.raise_for_status()
        update_data = update_response.json()
        
        if "errors" in update_data:
            logger.error(f"Hasura mutation error: {update_data['errors']}")
            return
        
        logger.info(f"✅ Updated tiling status for file_id={file_id}: '{status_value}'")
        
    except Exception as e:
        logger.error(f"❌ Hasura update failed for file_id={file_id}: {str(e)}")


# Job status tracking
class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# In-memory job store (for simple implementation)
jobs_store = {}
jobs_lock = threading.Lock()

# Initialize FastAPI app
app = FastAPI(
    title="Blocks Floorplan Tiler Service",
    description="High-quality PDF floorplan to tile pyramid converter for Leaflet/MapTiler",
    version="2.0.0"
)

# Request model
class ProcessFloorplanRequest(BaseModel):
    file_url: str
    file_id: int
    environment: str = "test"  # "test" or "production" - controls which storage account to use


class MassDeleteFloorplanRequest(BaseModel):
    file_ids: List[int]


class PdfAnnotationRequest(BaseModel):
    file_url: str
    metadata_url: str
    objects: List[Dict] = []
    environment: str = "test"


def pdf_to_images(pdf_content: bytes, scale: float = 2.0, max_dimension: int = 20000) -> List[Image.Image]:
    """
    Convert PDF bytes to a list of PIL Image objects using PyMuPDF (fitz).
    Optimized for large single-page floor plans with extreme aspect ratios.

    Args:
        pdf_content: PDF file content as bytes
        scale: Scale factor for rendering (higher = better quality)
               2.0 = 144 DPI (standard)
               4.0 = 288 DPI (high quality)
               6.0 = 432 DPI (very high quality)
               15.0 = 1080 DPI (extreme quality)
        max_dimension: Maximum width or height in pixels before reducing scale
                       This prevents timeouts on extremely large dimensions (default 20000)

    Returns:
        List of PIL Image objects, one per page
    """
    try:
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

        images = []

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            # Get page dimensions in points
            rect = page.rect
            width_pt = rect.width
            height_pt = rect.height

            # Calculate target dimensions
            target_width = int(width_pt * scale)
            target_height = int(height_pt * scale)

            # Check if dimensions are too large
            actual_scale = scale
            if target_width > max_dimension or target_height > max_dimension:
                logger.warning(f"Page {page_num + 1} dimensions too large ({target_width}x{target_height}), reducing scale")

                # Calculate reduced scale to fit within max_dimension
                scale_factor = max_dimension / max(target_width, target_height)
                actual_scale = scale * scale_factor

                target_width = int(width_pt * actual_scale)
                target_height = int(height_pt * actual_scale)

            # Create transformation matrix for scaling
            mat = fitz.Matrix(actual_scale, actual_scale)

            # Render page to pixmap (high quality)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Convert pixmap to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))

            images.append(pil_image)

        pdf_document.close()

        return images

    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise


class SimpleFloorplanTiler:
    """
    Simple CRS tiler for floorplans - treats image as flat 2D plane with pixel coordinates.
    Compatible with Leaflet's L.CRS.Simple (like MapTiler).
    Much simpler than Web Mercator - no geographic projection needed!
    """

    def __init__(self, tile_size: int = 256):
        self.tile_size = tile_size

    def tile_image(self,
                   source_img: Image.Image,
                   zoom_levels: List[int]) -> Dict[int, List[Tuple[int, int, Image.Image]]]:
        """
        Tile a floorplan image using Simple CRS (pixel coordinates)

        Args:
            source_img: PIL Image of the floorplan
            zoom_levels: List of zoom levels to generate (0 = most zoomed out)

        Returns:
            Dictionary mapping zoom level to list of (x, y, tile_image) tuples
        """
        if source_img.mode != 'RGBA':
            source_img = source_img.convert('RGBA')

        width, height = source_img.size
        max_zoom = max(zoom_levels)

        # Generate tiles for each zoom level
        pyramid = {}
        total_tiles = 0

        for zoom in sorted(zoom_levels, reverse=True):  # High to low
            tiles = self._generate_zoom_level(source_img, width, height, zoom, max_zoom)
            pyramid[zoom] = tiles
            tile_count = len(tiles)
            total_tiles += tile_count

        return pyramid

    def _generate_zoom_level(self,
                           source_img: Image.Image,
                           full_width: int,
                           full_height: int,
                           zoom: int,
                           max_zoom: int) -> List[Tuple[int, int, Image.Image]]:
        """Generate all tiles for a specific zoom level"""

        # Calculate scale factor for this zoom level
        # zoom 0 = most zoomed out, max_zoom = full resolution
        scale = 2 ** (zoom - max_zoom)

        # Calculate image dimensions at this zoom level
        scaled_width = int(full_width * scale)
        scaled_height = int(full_height * scale)

        # Resize image for this zoom level
        if zoom == max_zoom:
            scaled_img = source_img
        else:
            scaled_img = source_img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

        # Calculate number of tiles needed
        tiles_x = math.ceil(scaled_width / self.tile_size)
        tiles_y = math.ceil(scaled_height / self.tile_size)

        tiles = []

        # Generate each tile
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                tile_img = self._create_tile(scaled_img, tx, ty, scaled_width, scaled_height)
                if tile_img:
                    tiles.append((tx, ty, tile_img))

        # Release scaled image from memory if it's not the original
        if zoom != max_zoom:
            scaled_img.close()

        return tiles

    def _create_tile(self,
                    source_img: Image.Image,
                    tx: int, ty: int,
                    img_width: int, img_height: int) -> Image.Image:
        """Create a single tile from the source image"""

        # Calculate pixel bounds for this tile
        x1 = tx * self.tile_size
        y1 = ty * self.tile_size
        x2 = min(x1 + self.tile_size, img_width)
        y2 = min(y1 + self.tile_size, img_height)

        # Crop the tile from source image
        tile = source_img.crop((x1, y1, x2, y2))

        # If tile is smaller than tile_size (edge tiles), pad it with transparency
        if tile.width < self.tile_size or tile.height < self.tile_size:
            padded = Image.new('RGBA', (self.tile_size, self.tile_size), (0, 0, 0, 0))
            padded.paste(tile, (0, 0))
            return padded

        return tile


def generate_preview(image: Image.Image, max_width: int = 800) -> Image.Image:
    """
    Create low-res preview for initial load.

    Args:
        image: PIL Image to create preview from
        max_width: Maximum width for preview

    Returns:
        Preview image
    """
    ratio = image.width / image.height
    preview_width = min(max_width, image.width)
    preview_height = int(preview_width / ratio)
    return image.resize((preview_width, preview_height), Image.Resampling.LANCZOS)


def trim_whitespace(
    image: Image.Image,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    tolerance: int = 10,
    padding: int = 20
) -> Image.Image:
    """
    Auto-crop uniform background margins (e.g., white) from the image.
    Adds a small padding back to avoid cutting content too tightly.

    Args:
        image: PIL Image to crop
        bg_color: Background color to treat as whitespace (default white)
        tolerance: 0-255 threshold for considering pixels as background
        padding: Pixels to keep around detected content box

    Returns:
        Cropped PIL Image (or original if no content box found)
    """
    try:
        if image.mode != 'RGB':
            img = image.convert('RGB')
        else:
            img = image

        # Compute difference from a solid background image
        bg = Image.new('RGB', img.size, bg_color)
        diff = ImageChops.difference(img, bg).convert('L')
        # Threshold the difference to build a mask of non-background pixels
        mask = diff.point(lambda p: 255 if p > tolerance else 0)
        bbox = mask.getbbox()

        if not bbox:
            return image

        # Expand bbox by padding, clamped to image bounds
        left, top, right, bottom = bbox
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(img.width, right + padding)
        bottom = min(img.height, bottom + padding)

        if left == 0 and top == 0 and right == img.width and bottom == img.height:
            return image

        cropped = img.crop((left, top, right, bottom))
        return cropped
    except Exception as e:
        logger.warning(f"Whitespace trim failed: {e}")
        return image


def create_metadata(image: Image.Image, max_zoom: int, floorplan_id: str, tile_size: int, min_zoom: int = 0, zoom_levels: List[int] = None, file_id: int = None) -> dict:
    """
    Generate metadata for Web Mercator tile consumption.

    Args:
        image: PIL Image
        max_zoom: Maximum zoom level
        floorplan_id: Unique identifier for this floor plan
        tile_size: Size of tiles in pixels
        min_zoom: Minimum zoom level
        zoom_levels: List of actual zoom levels generated
        file_id: File ID (required)

    Returns:
        Metadata dictionary compatible with Web Mercator tiling
    """
    # Simple CRS bounds - just pixel coordinates (like MapTiler)
    # No geographic projection needed!
    image_bounds = [[0, 0], [image.height, image.width]]  # [[y_min, x_min], [y_max, x_max]]

    return {
        "floorplan_id": floorplan_id,
        "file_id": file_id,
        "source_image": {
            "width": image.width,
            "height": image.height,
            "format": "RGBA"
        },
        "tile_size": tile_size,
        "max_zoom": max_zoom,
        "min_zoom": max(0, min_zoom),
        "zoom_levels": zoom_levels or list(range(max(0, min_zoom), max_zoom + 1)),
        "bounds": image_bounds,  # Simple pixel coordinate bounds
        "coordinate_system": "Simple CRS (L.CRS.Simple) - pixel coordinates, compatible with MapTiler",
        "center": [image.height / 2, image.width / 2],  # Center in pixel coordinates [y, x]
        "created_at": datetime.utcnow().isoformat(),
        "tile_format": "png",
        "total_tiles": None,  # Will be filled by caller
        "usage_notes": {
            "leaflet_crs": "Use L.CRS.Simple for flat floorplan display",
            "tile_url_template": "{baseUrl}/{z}/{x}/{y}.png",
            "bounds_format": "Geographic coordinates (lat/lon)"
        }
    }


def upload_tiles_to_blob(
    pyramid: Dict[int, List[Tuple[int, int, Image.Image]]],
    preview: Image.Image,
    metadata: dict,
    floorplan_id: str,
    original_blob_name: str,
    connection_string: str,
    container: str = "blocks",
    base_image: Image.Image | None = None,
    base_image_data: bytes | None = None,
    base_image_format: str = "png"
):
    """
    Upload entire tile pyramid to Azure Blob Storage.
    Structure: blocks/floorplans/{floorplan-id}/
                   ├── metadata.json
                   ├── preview.jpg
                   ├── base.png (full-resolution rendered image used for tiling) [optional]
                   └── tiles/{z}/{x}/{y}.png

    Args:
        pyramid: Dictionary of zoom level to tiles
        preview: Preview image
        metadata: Metadata dictionary
        floorplan_id: Unique identifier
        original_blob_name: Original blob name
        connection_string: Azure Storage connection string
        container: Container name (default: "blocks")
    """
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container)

    # Container should already exist (floor-plans)
    try:
        container_client.get_container_properties()
    except Exception as e:
        logger.warning(f"Container check failed: {str(e)}")

    # Upload metadata
    metadata_blob = f"floorplans/{floorplan_id}/metadata.json"
    container_client.upload_blob(
        metadata_blob,
        json.dumps(metadata, indent=2),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json")
    )

    # Upload preview
    preview_bytes = io.BytesIO()
    preview.save(preview_bytes, format='JPEG', quality=85, optimize=True)
    preview_bytes.seek(0)
    container_client.upload_blob(
        f"floorplans/{floorplan_id}/preview.jpg",
        preview_bytes,
        overwrite=True,
        content_settings=ContentSettings(content_type="image/jpeg")
    )

    # Upload optimized base image (optional)
    if base_image_data is not None:
        try:
            base_filename = f"base-image.{base_image_format}"
            content_type = f"image/{base_image_format}" if base_image_format != "webp" else "image/webp"

            container_client.upload_blob(
                f"floorplans/{floorplan_id}/{base_filename}",
                base_image_data,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type)
            )
            size_mb = len(base_image_data) / (1024*1024)
        except Exception as be:
            logger.warning(f"Failed to upload base image: {be}")

    # Upload all tiles
    total_tiles = sum(len(tiles) for tiles in pyramid.values())
    uploaded = 0

    for zoom, tiles in pyramid.items():
        for x, y, tile_image in tiles:
            tile_bytes = io.BytesIO()
            # Use balanced compression (6) for good quality with smaller files
            tile_image.save(tile_bytes, format='PNG', compress_level=6)
            tile_bytes.seek(0)

            # Leaflet standard: {z}/{x}/{y}.png
            blob_path = f"floorplans/{floorplan_id}/tiles/{zoom}/{x}/{y}.png"
            container_client.upload_blob(
                blob_path,
                tile_bytes,
                overwrite=True,
                content_settings=ContentSettings(content_type="image/png")
            )

            uploaded += 1


def extract_floorplan_id(blob_name: str) -> str:
    """
    Extract floor plan ID from blob name.
    Example: floor-plans/myplan.pdf -> myplan
    """
    filename = blob_name.split('/')[-1]
    return filename.rsplit('.', 1)[0]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Blocks Floorplan Tiler Service",
        "status": "running",
        "version": "2.0.0",
        "endpoint": "/api/process-floorplan"
    }


@app.get("/health")
async def health():
    """Kubernetes/Container Apps health probe"""
    return {"status": "healthy"}


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str, api_key_valid: bool = Depends(verify_api_key)):
    """Get the status of a processing job"""
    with jobs_lock:
        if job_id not in jobs_store:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs_store[job_id]
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "result": job.get("result")
        }


@app.delete("/api/delete-floorplan/{file_id}")
async def delete_floorplan(file_id: int, api_key_valid: bool = Depends(verify_api_key)):
    """
    Delete all floorplans matching the given file_id.

    Args:
        file_id: File ID as path parameter

    Returns:
        JSON response with deletion status
    """
    try:
        # Get storage connection
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise HTTPException(
                status_code=500,
                detail="Storage connection string not configured"
            )

        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service.get_container_client("blocks")

        # Find all blobs matching the pattern
        prefix = f"floorplans/{file_id}/"

        existing_blobs = container_client.list_blobs(name_starts_with=prefix)
        blobs_to_delete = []

        for blob in existing_blobs:
            blobs_to_delete.append(blob.name)

        if not blobs_to_delete:
            return {
                "success": True,
                "message": "No floorplans found to delete",
                "file_id": file_id,
                "deleted_count": 0
            }

        # Delete all matching blobs
        deleted_count = 0
        failed_deletions = []

        for blob_name in blobs_to_delete:
            try:
                container_client.delete_blob(blob_name)
                deleted_count += 1
            except Exception as del_err:
                logger.error(f"Failed to delete {blob_name}: {del_err}")
                failed_deletions.append(blob_name)

        if failed_deletions:
            logger.warning(f"Failed to delete {len(failed_deletions)} blobs")
            return {
                "success": False,
                "message": f"Partially deleted. {deleted_count} succeeded, {len(failed_deletions)} failed",
                "file_id": file_id,
                "deleted_count": deleted_count,
                "failed_count": len(failed_deletions)
            }

        return {
            "success": True,
            "message": "All floorplans deleted successfully",
            "file_id": file_id,
            "deleted_count": deleted_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting floorplan: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting floorplan: {str(e)}"
        )


@app.delete("/api/mass-delete-floorplan")
async def mass_delete_floorplan(request: MassDeleteFloorplanRequest, api_key_valid: bool = Depends(verify_api_key)):
    """
    Delete multiple floorplans in a single request.

    Args:
        request: MassDeleteFloorplanRequest with array of file_ids

    Returns:
        JSON response with deletion results for each item
    """
    try:
        # Get storage connection
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise HTTPException(
                status_code=500,
                detail="Storage connection string not configured"
            )

        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service.get_container_client("blocks")

        results = []
        total_deleted = 0

        for file_id in request.file_ids:
            try:
                # Find all blobs matching the pattern
                prefix = f"floorplans/{file_id}/"

                existing_blobs = container_client.list_blobs(name_starts_with=prefix)
                blobs_to_delete = [blob.name for blob in existing_blobs]

                if not blobs_to_delete:
                    results.append({
                        "file_id": file_id,
                        "success": True,
                        "deleted_count": 0,
                        "message": "No floorplans found"
                    })
                    continue

                # Delete all matching blobs
                deleted_count = 0
                failed_deletions = []

                for blob_name in blobs_to_delete:
                    try:
                        container_client.delete_blob(blob_name)
                        deleted_count += 1
                        total_deleted += 1
                    except Exception as del_err:
                        logger.error(f"Failed to delete {blob_name}: {del_err}")
                        failed_deletions.append(blob_name)

                if failed_deletions:
                    results.append({
                        "file_id": file_id,
                        "success": False,
                        "deleted_count": deleted_count,
                        "failed_count": len(failed_deletions),
                        "message": f"Partially deleted: {deleted_count} succeeded, {len(failed_deletions)} failed"
                    })
                else:
                    results.append({
                        "file_id": file_id,
                        "success": True,
                        "deleted_count": deleted_count,
                        "message": "All floorplans deleted successfully"
                    })

            except Exception as item_err:
                logger.error(f"Error deleting {file_id}: {str(item_err)}")
                results.append({
                    "file_id": file_id,
                    "success": False,
                    "deleted_count": 0,
                    "message": f"Error: {str(item_err)}"
                })

        # Summary
        successful_items = sum(1 for r in results if r["success"])
        failed_items = len(results) - successful_items

        return {
            "success": failed_items == 0,
            "message": f"Mass deletion completed: {successful_items} successful, {failed_items} failed",
            "total_items": len(results),
            "successful_items": successful_items,
            "failed_items": failed_items,
            "total_blobs_deleted": total_deleted,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in mass deletion: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error in mass deletion: {str(e)}"
        )


def update_job_progress(job_id: str, progress: int, message: str):
    """Helper to update job progress"""
    with jobs_lock:
        if job_id in jobs_store:
            jobs_store[job_id]["progress"] = progress
            jobs_store[job_id]["message"] = message
            jobs_store[job_id]["updated_at"] = datetime.utcnow().isoformat()


def process_floorplan_background(job_id: str, file_url: str, file_id: int, environment: str = "test"):
    """Background task to process the floorplan"""
    try:
        # Update status to processing
        with jobs_lock:
            jobs_store[job_id]["status"] = JobStatus.PROCESSING
            jobs_store[job_id]["updated_at"] = datetime.utcnow().isoformat()
        
        # 1️⃣ START: Update Hasura status
        update_tiling_status(file_id, "Tiling running")

        # Call the synchronous processing logic
        result = process_floorplan_sync(file_url, job_id, file_id, environment)

        # Mark as completed
        with jobs_lock:
            jobs_store[job_id]["status"] = JobStatus.COMPLETED
            jobs_store[job_id]["progress"] = 100
            jobs_store[job_id]["updated_at"] = datetime.utcnow().isoformat()
            jobs_store[job_id]["message"] = "Processing completed successfully"
            jobs_store[job_id]["result"] = result
        
        # 2️⃣ SUCCESS: Update Hasura status
        update_tiling_status(file_id, "Tiling finished")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}", exc_info=True)
        with jobs_lock:
            jobs_store[job_id]["status"] = JobStatus.FAILED
            jobs_store[job_id]["updated_at"] = datetime.utcnow().isoformat()
            jobs_store[job_id]["message"] = str(e)
        
        # 3️⃣ ERROR: Update Hasura status
        update_tiling_status(file_id, "Error", error_message=str(e))


@app.post("/api/process-floorplan")
async def process_floorplan(request: ProcessFloorplanRequest, background_tasks: BackgroundTasks, api_key_valid: bool = Depends(verify_api_key)):
    """
    Submit a PDF floorplan for processing (async - returns immediately).

    Args:
        request: ProcessFloorplanRequest with file_url

    Returns:
        JSON response with job_id for tracking progress
    """
    # Validate file URL
    if not request.file_url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid file URL - must be http:// or https://")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Create job record
    now = datetime.utcnow().isoformat()
    with jobs_lock:
        jobs_store[job_id] = {
            "job_id": job_id,
            "file_url": request.file_url,
            "status": JobStatus.QUEUED,
            "progress": 0,
            "message": "Job queued for processing",
            "created_at": now,
            "updated_at": now,
            "result": None
        }

    # Start background processing
    background_tasks.add_task(process_floorplan_background, job_id, request.file_url, request.file_id, request.environment)

    return {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "message": "Job queued for processing",
        "status_url": f"/api/status/{job_id}"
    }


def process_floorplan_sync(file_url: str, job_id: str, file_id: int, environment: str = "test"):
    """
    Synchronous processing logic (moved from original async endpoint).
    Updates job progress throughout processing.

    Args:
        file_url: URL of the PDF to process
        job_id: Job ID for progress tracking
        file_id: File ID (required)
        environment: "test" or "production" - determines which storage account to use

    Returns:
        Result dictionary with floorplan info
    """
    update_job_progress(job_id, 5, "Downloading PDF...")

    # Determine storage configuration based on environment
    if environment.lower() == "production":
        if not PRODUCTION_STORAGE_CONNECTION_STRING or not PRODUCTION_STORAGE_ACCOUNT_NAME:
            raise HTTPException(
                status_code=500,
                detail="Production storage not configured. Please set PRODUCTION_STORAGE_CONNECTION_STRING and PRODUCTION_STORAGE_ACCOUNT_NAME environment variables."
            )
        connection_string = PRODUCTION_STORAGE_CONNECTION_STRING
        storage_account_name = PRODUCTION_STORAGE_ACCOUNT_NAME
    else:
        if not TEST_STORAGE_CONNECTION_STRING:
            raise HTTPException(
                status_code=500,
                detail="Test storage connection string not configured"
            )
        connection_string = TEST_STORAGE_CONNECTION_STRING
        storage_account_name = TEST_STORAGE_ACCOUNT_NAME

    try:
        # Download the PDF from Azure Blob Storage or URL
        try:
            # Check if it's an Azure Blob Storage URL
            if 'blob.core.windows.net' in file_url:
                # Parse the blob URL to extract storage account, container and blob name
                from urllib.parse import urlparse
                parsed_url = urlparse(file_url)
                # URL format: https://<account>.blob.core.windows.net/<container>/<blob-path>
                source_storage_account = parsed_url.hostname.split('.')[0] if parsed_url.hostname else ''
                path_parts = parsed_url.path.lstrip('/').split('/', 1)
                container_name = path_parts[0]
                blob_name = path_parts[1] if len(path_parts) > 1 else ''

                # Determine which connection string to use based on source storage account
                if source_storage_account.lower() == PRODUCTION_STORAGE_ACCOUNT_NAME.lower() if PRODUCTION_STORAGE_ACCOUNT_NAME else False:
                    download_connection_string = PRODUCTION_STORAGE_CONNECTION_STRING
                else:
                    download_connection_string = TEST_STORAGE_CONNECTION_STRING

                if not download_connection_string:
                    raise Exception(f"Storage connection string not configured for source account: {source_storage_account}")

                blob_service = BlobServiceClient.from_connection_string(download_connection_string)
                blob_client = blob_service.get_blob_client(container_name, blob_name)
                file_content = blob_client.download_blob().readall()
                update_job_progress(job_id, 10, "PDF downloaded, starting conversion...")
            else:
                # Download from external URL
                import urllib.request
                with urllib.request.urlopen(file_url) as response:
                    file_content = response.read()
                update_job_progress(job_id, 10, "PDF downloaded, starting conversion...")
        except Exception as download_error:
            logger.error(f"Error downloading file: {str(download_error)}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download file: {str(download_error)}"
            )

        # Validate PDF size to prevent OOM crashes
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > 100:
            raise HTTPException(
                status_code=400,
                detail=f"PDF file too large ({file_size_mb:.1f}MB). Maximum allowed size is 100MB."
            )

        # Extract filename from URL
        from urllib.parse import urlparse
        parsed_url = urlparse(file_url)
        floorplan_name = parsed_url.path.split('/')[-1]
        if not floorplan_name.lower().endswith('.pdf'):
            floorplan_name = 'floorplan.pdf'

        # Extract base name from filename (remove .pdf extension)
        base_name = floorplan_name.rsplit('.', 1)[0]

        # Create floorplan ID - just use file_id as folder name
        floorplan_id = str(file_id)

        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service.get_container_client("blocks")

        # Check if any floorplan already exists with this file_id
        prefix = f"floorplans/{file_id}/"
        existing_blob = None

        # Get iterator and check if any blob exists with this prefix
        blob_iterator = container_client.list_blobs(name_starts_with=prefix)
        try:
            existing_blob = next(iter(blob_iterator), None)
        except Exception as e:
            logger.warning(f"Error checking for existing blobs: {e}")

        if existing_blob:
            # Found existing floorplan with this file_id, skip processing
            logger.info(f"Floorplan already exists for file_id {file_id}, skipping processing")

            # Extract the existing floorplan_id from the first blob path
            existing_blob_path = existing_blob.name
            existing_floorplan_id = existing_blob_path.split('/')[1]

            return {
                "success": True,
                "message": "Floorplan already exists for this file_id",
                "floorplan_id": existing_floorplan_id,
                "environment": environment,
                "urls": {
                    "metadata": f"https://{storage_account_name}.blob.core.windows.net/blocks/floorplans/{existing_floorplan_id}/metadata.json",
                    "preview": f"https://{storage_account_name}.blob.core.windows.net/blocks/floorplans/{existing_floorplan_id}/preview.jpg",
                    "tiles": f"https://{storage_account_name}.blob.core.windows.net/blocks/floorplans/{existing_floorplan_id}/tiles/{{z}}/{{x}}/{{y}}.png"
                }
            }

        # Create a mock blob name for compatibility
        myblob_name = f"blocks/{floorplan_name}"

        # ============================================================
        # 🚀 PRODUCTION MODE - HIGH QUALITY TILING SETTINGS
        # ============================================================

        # 🎨 QUALITY SETTINGS (Configurable via environment variables):
        PDF_SCALE = float(os.environ.get('PDF_SCALE', '40.0'))  # 40.0=2880 DPI (extreme quality for deep zoom)
        MAX_DIMENSION = int(os.environ.get('MAX_DIMENSION', '30000'))  # 30K pixels max

        # 🗺️ TILING CONFIGURATION - DEEP ZOOM MODE:
        MAX_ZOOM_LIMIT = 12      # Maximum zoom levels allowed
        FORCED_MAX_Z_ENV = int(os.environ.get('FORCED_MAX_Z', '-1'))  # -1 = auto-calculate based on image size
        ZOOM_BOOST = int(os.environ.get('ZOOM_BOOST', '4'))  # Add extra zoom levels beyond native (for deep zoom with upscaling)
        TILE_SIZE_ENV = int(os.environ.get('TILE_SIZE', '512'))  # 512px tiles
        MIN_ZOOM_ENV = int(os.environ.get('MIN_ZOOM', '0'))  # Start from zoom level 0

        # ✅ Deep zoom with upscaling beyond native resolution
        # For PDF_SCALE=15: Native res at zoom ~6-7, upscales for 8-10
        # For PDF_SCALE=40: Native res at zoom ~8-9, upscales for 10-12
        # ============================================================

        # Check if the file is a PDF
        if not floorplan_name.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="File must be a PDF"
            )

        # Guard: Only process PDFs uploaded at the container root to avoid re-processing
        try:
            container_and_rest = myblob_name.split('/', 1)
            relative_path = container_and_rest[1] if len(container_and_rest) > 1 else myblob_name
            if '/' in relative_path:
                return JSONResponse(
                    content={"success": True, "message": "Skipped nested PDF to prevent re-processing"},
                    status_code=200
                )
        except Exception as _:
            pass

        # Validate TILE_SIZE to supported values
        if TILE_SIZE_ENV not in (128, 256, 512, 1024):
            logger.warning(f"Unsupported TILE_SIZE={TILE_SIZE_ENV}; defaulting to 256")
            TILE_SIZE_ENV = 256

        # SMART SCALING: Analyze PDF characteristics and determine optimal scale
        # Check file size - simple documents are small, complex floorplans are large
        file_size_mb = len(file_content) / (1024 * 1024)

        # Open PDF to check dimensions and calculate appropriate quality
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        if pdf_document.page_count > 0:
            page = pdf_document[0]
            rect = page.rect
            width_pt = rect.width
            height_pt = rect.height

            # Convert points to inches (72 points = 1 inch)
            width_inches = width_pt / 72.0
            height_inches = height_pt / 72.0
            max_dimension_inches = max(width_inches, height_inches)

            # Calculate aspect ratio
            aspect_ratio = max(width_inches, height_inches) / min(width_inches, height_inches)

            # Quick content analysis: Check if page has embedded images (raster content)
            has_images = False
            image_count = 0
            try:
                image_list = page.get_images(full=False)
                image_count = len(image_list)
                has_images = image_count > 0
            except Exception:
                pass

            # File size heuristic: Small files are simple documents, large files are complex plans
            is_complex_plan = file_size_mb > 0.5  # Files over 500KB are likely detailed plans

            # INTELLIGENT SCALE SELECTION based on document characteristics
            # Large architectural plans need high DPI for deep zoom
            # Standard documents need moderate DPI for readability

            # Large architectural drawings (3+ feet)
            # Use high scale for deep zoom capability
            if max_dimension_inches >= 36:
                target_scale = PDF_SCALE  # Keep 40x (2880 DPI)
                reason = "large architectural plan (36+ inches)"

            # Medium architectural drawings (2-3 feet)
            elif max_dimension_inches >= 24:
                target_scale = min(PDF_SCALE, 30.0)  # Up to 30x (2160 DPI)
                reason = "medium architectural plan (24-36 inches)"

            # Large format documents (tabloid/A3)
            # Boost scale for complex/detailed plans (large file size)
            elif max_dimension_inches >= 17:
                if has_images:
                    target_scale = min(PDF_SCALE, 15.0)  # 15x (1080 DPI) for scans
                else:
                    # Complex vector plans get higher DPI
                    target_scale = min(PDF_SCALE, 40.0 if is_complex_plan else 20.0)
                reason = "large format document (17-24 inches)"

            # Standard documents (letter/A4)
            # Complex plans need more detail than simple documents
            elif max_dimension_inches >= 11:
                if has_images:
                    target_scale = min(PDF_SCALE, 12.0)  # 12x (864 DPI) for scans
                else:
                    # Boost for complex vector plans with lots of detail
                    target_scale = min(PDF_SCALE, 35.0 if is_complex_plan else 15.0)
                reason = "standard document (11-17 inches)"

            # Small documents
            else:
                target_scale = min(PDF_SCALE, 10.0)  # Up to 10x (720 DPI)
                reason = "small document (<11 inches)"

            # Content-based adjustment
            if has_images:
                reason += ", scanned/raster content"
            else:
                reason += ", vector/CAD content"

            if is_complex_plan and max_dimension_inches < 36:
                reason += " (complex/detailed)"

            # Additional check: Extremely wide/tall aspect ratios (like long floorplans)
            # need higher quality even if overall size is moderate
            if aspect_ratio > 2.5 and max_dimension_inches >= 20:
                target_scale = min(target_scale * 1.5, PDF_SCALE)
                reason += " with extreme aspect ratio"

            # Final safety check: Don't exceed memory limits
            potential_width = int(width_pt * target_scale)
            potential_height = int(height_pt * target_scale)
            potential_pixels = potential_width * potential_height

            MAX_SAFE_PIXELS = 300_000_000  # ~17,000 × 17,000 pixels
            if potential_pixels > MAX_SAFE_PIXELS:
                area_scale = (MAX_SAFE_PIXELS / potential_pixels) ** 0.5
                target_scale = target_scale * area_scale
                reason += " (reduced for memory safety)"

            PDF_SCALE = target_scale

        pdf_document.close()

        # 1. Convert PDF to PNG (single page expected)
        update_job_progress(job_id, 15, "Converting PDF to image...")
        images = pdf_to_images(file_content, scale=PDF_SCALE, max_dimension=MAX_DIMENSION)

        if len(images) == 0:
            raise Exception("No images generated from PDF")

        # Use first page (floor plans should be single page)
        floor_plan_image = images[0]
        update_job_progress(job_id, 20, "PDF converted, preparing image...")

        # Release extra images from memory immediately
        if len(images) > 1:
            for img in images[1:]:
                img.close()
            images = [floor_plan_image]

        # Optional: auto-trim white margins
        original_image = floor_plan_image
        floor_plan_image = trim_whitespace(floor_plan_image, bg_color=(255, 255, 255), tolerance=10, padding=20)
        # Release original image if trim created a new one
        if original_image is not floor_plan_image:
            original_image.close()

        # 2. Calculate optimal zoom levels for Leaflet tiles
        tile_size = TILE_SIZE_ENV

        # Calculate optimal max zoom based on image dimensions
        # Goal: At max zoom, we want roughly 1-4 tiles per dimension (perfect native resolution)
        if FORCED_MAX_Z_ENV == -1:
            # Auto-calculate: Find zoom level where image fits in ~2-8 tiles per dimension
            max_dim = max(floor_plan_image.width, floor_plan_image.height)

            # Calculate tiles needed at zoom 0 (1 tile covers entire image)
            # At each zoom level, tiles double: zoom 0 = 1 tile, zoom 1 = 2 tiles, zoom 2 = 4 tiles, etc.
            # We want: tile_size * (2^zoom) ≈ max_dim
            # So: 2^zoom ≈ max_dim / tile_size
            # zoom ≈ log2(max_dim / tile_size)

            optimal_zoom = math.ceil(math.log2(max_dim / tile_size))

            # Adaptive zoom boost based on image size
            # Large images benefit from deep zoom, small images don't need excessive upscaling
            if max_dim > 20000:
                adaptive_boost = ZOOM_BOOST  # Full boost for very large plans
            elif max_dim > 10000:
                adaptive_boost = max(3, ZOOM_BOOST - 1)  # Moderate boost
            else:
                adaptive_boost = max(2, ZOOM_BOOST - 2)  # Minimal boost for small images
            
            boosted_zoom = optimal_zoom + adaptive_boost
            max_zoom = max(0, min(boosted_zoom, MAX_ZOOM_LIMIT))
        else:
            # Use forced max zoom from environment
            max_zoom = max(0, min(FORCED_MAX_Z_ENV, MAX_ZOOM_LIMIT))

        min_zoom = max(0, min(MIN_ZOOM_ENV, max_zoom))
        total_levels = (max_zoom - min_zoom + 1)

        # 3. Generate tile pyramid using Simple CRS
        update_job_progress(job_id, 30, f"Generating {total_levels} zoom levels of tiles...")

        zoom_levels = list(range(min_zoom, max_zoom + 1))
        floorplan_tiler = SimpleFloorplanTiler(tile_size=tile_size)
        pyramid = floorplan_tiler.tile_image(floor_plan_image, zoom_levels)
        total_tiles = sum(len(tiles) for tiles in pyramid.values())
        update_job_progress(job_id, 60, f"Generated {total_tiles} tiles, uploading to storage...")

        # 4. Generate preview image
        preview = generate_preview(floor_plan_image, max_width=800)

        # Skip base image to save memory - tiles are sufficient

        # 5. Create metadata
        metadata = create_metadata(
            floor_plan_image,
            max_zoom,
            floorplan_id,
            tile_size=tile_size,
            min_zoom=min_zoom,
            zoom_levels=zoom_levels,
            file_id=file_id
        )
        metadata["total_tiles"] = total_tiles
        metadata["quality_settings"] = {
            "pdf_scale": PDF_SCALE,
            "max_dimension": MAX_DIMENSION,
            "dpi": PDF_SCALE * 72
        }

        # 6. Upload to blob storage
        upload_tiles_to_blob(
            pyramid=pyramid,
            preview=preview,
            metadata=metadata,
            floorplan_id=floorplan_id,
            original_blob_name=myblob_name,
            connection_string=connection_string,
            container="blocks",
            base_image=None,
            base_image_data=None,
            base_image_format="png"
        )
        update_job_progress(job_id, 90, f"Uploaded {total_tiles} tiles to storage...")

        # Release memory
        preview.close()
        for zoom_tiles in pyramid.values():
            for _, _, tile_img in zoom_tiles:
                tile_img.close()
        pyramid.clear()

        # 7. Archive the original PDF
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        source_container = "blocks"

        try:
            dest_blob_name = f"floorplans/{floorplan_id}/{floorplan_id}.pdf"
            dest_client = blob_service.get_blob_client(source_container, dest_blob_name)
            content_settings = ContentSettings(content_type="application/pdf")
            dest_client.upload_blob(file_content, overwrite=True, content_settings=content_settings)
        except Exception as copy_err:
            logger.warning(f"Could not store archived PDF copy: {str(copy_err)}")

        update_job_progress(job_id, 95, "Finalizing floor plan processing...")

        # Return success response
        return {
            "success": True,
            "floorplan_id": floorplan_id,
            "environment": environment,
            "storage_account": storage_account_name,
            "dimensions": {
                "width": floor_plan_image.width,
                "height": floor_plan_image.height
            },
            "quality": {
                "dpi": int(PDF_SCALE * 72)
            },
            "tiles": {
                "total": total_tiles,
                "zoom_levels": total_levels,
                "min_zoom": min_zoom,
                "max_zoom": max_zoom,
                "tile_size": tile_size
            },
            "urls": {
                "metadata": f"https://{storage_account_name}.blob.core.windows.net/blocks/floorplans/{floorplan_id}/metadata.json",
                "preview": f"https://{storage_account_name}.blob.core.windows.net/blocks/floorplans/{floorplan_id}/preview.png",
                "tiles": f"https://{storage_account_name}.blob.core.windows.net/blocks/floorplans/{floorplan_id}/tiles/{{z}}/{{x}}/{{y}}.png"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


import pdf_annotation


@app.post("/api/pdf-annotation")
async def pdf_annotation_endpoint(request: PdfAnnotationRequest,
                                  api_key_valid: bool = Depends(verify_api_key)):
    """
    Annotate a PDF with shapes and markers from Leaflet drawings.

    Request body:
    {
        "file_url": "https://example.com/floorplan.pdf",
        "metadata_url": "https://example.com/floorplan/metadata.json",
        "objects": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[lat, lon], ...]]},
                "properties": {"type": "rectangle"}
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lat, lon]},
                "properties": {"type": "marker", "content": "Label"}
            }
        ],
        "environment": "test"
    }

    Returns:
    {
        "success": true,
        "annotated_pdf_url": "https://...",
        "filename": "floorplan-annotation-[timestamp].pdf"
    }
    """
    try:
        # Validate required fields
        file_url = request.file_url
        metadata_url = request.metadata_url
        objects = request.objects or []

        if not file_url:
            raise HTTPException(status_code=400, detail="file_url is required")

        if not metadata_url:
            raise HTTPException(status_code=400, detail="metadata_url is required")

        logger.info(f"📝 Starting PDF annotation")
        logger.info(f"   PDF URL: {file_url}")
        logger.info(f"   Metadata URL: {metadata_url}")
        logger.info(f"   Objects to draw: {len(objects)}")

        # Determine storage configuration based on environment
        environment = request.environment
        if environment == "production" and PRODUCTION_STORAGE_CONNECTION_STRING:
            connection_string = PRODUCTION_STORAGE_CONNECTION_STRING
            storage_account_name = PRODUCTION_STORAGE_ACCOUNT_NAME
        else:
            connection_string = TEST_STORAGE_CONNECTION_STRING
            storage_account_name = TEST_STORAGE_ACCOUNT_NAME

        if not connection_string:
            raise HTTPException(status_code=500, detail="Azure Storage connection string not configured")

        # Download metadata
        logger.info("⬇️ Downloading metadata...")
        metadata_bytes = await pdf_annotation.download_file(metadata_url)
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        logger.info(f"✅ Metadata loaded: {metadata.get('floorplan_id')}")

        # Download PDF
        logger.info("⬇️ Downloading PDF...")
        pdf_bytes = await pdf_annotation.download_file(file_url)
        logger.info(f"✅ PDF downloaded: {len(pdf_bytes)} bytes")

        # Annotate PDF
        logger.info("🎨 Annotating PDF...")
        annotated_pdf_bytes = pdf_annotation.annotate_pdf(pdf_bytes, objects, metadata)
        logger.info(f"✅ PDF annotated: {len(annotated_pdf_bytes)} bytes")

        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        # Extract original filename without extension
        original_filename = file_url.split("/")[-1].rsplit(".", 1)[0]
        annotated_filename = f"{original_filename}-annotation-{timestamp}.pdf"

        # Upload to Azure Blob Storage
        logger.info("☁️ Uploading to Azure Blob Storage...")
        blob_service = BlobServiceClient.from_connection_string(connection_string)

        # Upload to 'annotated-pdfs' container
        container_name = "annotated-pdfs"
        try:
            container_client = blob_service.get_container_client(container_name)
            container_client.get_container_properties()
        except Exception:
            # Create container if it doesn't exist
            container_client = blob_service.create_container(container_name, public_access="blob")
            logger.info(f"Created container: {container_name}")

        # Upload the annotated PDF
        blob_client = blob_service.get_blob_client(container_name, annotated_filename)
        blob_client.upload_blob(
            annotated_pdf_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/pdf")
        )

        # Generate the public URL
        annotated_pdf_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{annotated_filename}"

        logger.info(f"✅ Upload complete!")
        logger.info(f"   URL: {annotated_pdf_url}")

        # Return success response
        return {
            "success": True,
            "annotated_pdf_url": annotated_pdf_url,
            "filename": annotated_filename,
            "objects_drawn": len(objects),
            "metadata": {
                "floorplan_id": metadata.get("floorplan_id"),
                "source_url": file_url
            }
        }

    except httpx.HTTPError as e:
        logger.error(f"❌ Download error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error annotating PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error annotating PDF: {str(e)}")


if __name__ == "__main__":
    # For local development
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
