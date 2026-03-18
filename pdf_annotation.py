"""
PDF Annotation Module
Handles annotating PDFs with shapes and markers.
"""

import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import azure.functions as func
import fitz  # PyMuPDF for PDF annotation
import httpx
from azure.storage.blob import BlobServiceClient, ContentSettings
from PIL import Image, ImageChops

# ==========================================
# PDF ANNOTATION CONFIGURATION
# ==========================================

# Annotation styling
ANNOTATION_CONFIG = {
    "polygon": {
        "fill_color": (1, 0, 0),      # Red (RGB normalized 0-1)
        "fill_opacity": 0.2,
        "stroke_color": (1, 0, 0),
        "stroke_width": 1,
        "stroke_opacity": 1.0
    },
    "marker": {
        "fill_color": (1, 0, 0),
        "fill_opacity": 0.2,
        "radius": 4,
        "stroke_color": (0, 0, 0),
        "stroke_width": 1
    },
    "square": {
        "fill_color": (1, 0, 0),
        "fill_opacity": 0.25,
        "stroke_color": (1, 0, 0),
        "stroke_width": 1,
        "stroke_opacity": 1.0
    },
    "text": {
        "font_size": 14,
        "font_color": (1, 0, 0),
        "background_color": (1, 1, 1),
        "background_opacity": 0.9,
        "padding": 4
    }
}


# ==========================================
# COORDINATE TRANSFORMATION
# ==========================================
#
# Forward pipeline (tile generation in app.py):
#   1. PDF → Image:  pixel = pdf_pt × pdf_scale        (fitz.Matrix)
#   2. Trim whitespace: crops margins, shifts origin     (trim_whitespace)
#   3. Image → Leaflet:  lng = pixel_x / 2^maxZoom      (CRS.Simple)
#                         lat = -pixel_y / 2^maxZoom
#
# Reverse pipeline (what we need for annotations):
#   1. Leaflet → trimmed image pixels
#   2. Add trim offset → original (pre-trim) image pixels
#   3. Divide by pdf_scale → PDF points
#
# Formula:
#   pdf_x = (leaflet_x × 2^maxZoom + trim_left) / pdf_scale
#   pdf_y = (-leaflet_y × 2^maxZoom + trim_top) / pdf_scale
#
# When no trimming occurred: trim_left=0, trim_top=0, simplifies to old formula
# ==========================================

def detect_trim_offset(page: fitz.Page, metadata: Dict[str, Any]) -> Tuple[float, float]:
    """
    Detect the whitespace trim offset by comparing PDF page dimensions
    with the stored image dimensions. If trimming occurred, re-render at
    low resolution to find the exact content origin.

    Uses the same logic as trim_whitespace() in app.py:
    - bg_color=(255,255,255), tolerance=10, padding=20 pixels

    Args:
        page: PyMuPDF page object (original PDF)
        metadata: Metadata with source_image dimensions and pdf_scale

    Returns:
        (trim_left, trim_top) in pixels at pdf_scale resolution.
        These are the pixel offsets that were cropped from the pre-trim image.
    """
    pdf_scale = metadata["quality_settings"]["pdf_scale"]
    img_w = metadata["source_image"]["width"]
    img_h = metadata["source_image"]["height"]

    # Pre-trim image dimensions
    pretrim_w = page.rect.width * pdf_scale
    pretrim_h = page.rect.height * pdf_scale

    # Check if trimming occurred (allow 1px tolerance for rounding)
    if abs(pretrim_w - img_w) <= 1 and abs(pretrim_h - img_h) <= 1:
        logging.info("No whitespace trimming detected — using direct formula")
        return (0.0, 0.0)

    logging.info(f"Whitespace trimming detected!")
    logging.info(f"  Pre-trim:  {pretrim_w:.0f} x {pretrim_h:.0f} px")
    logging.info(f"  Post-trim: {img_w} x {img_h} px")
    logging.info(f"  Trimmed:   {pretrim_w - img_w:.0f} x {pretrim_h - img_h:.0f} px")

    # Re-render at low resolution to detect content bbox
    # Use scale 2.0 (144 DPI) — fast and accurate enough for bbox detection
    detect_scale = 2.0
    pix = page.get_pixmap(matrix=fitz.Matrix(detect_scale, detect_scale), alpha=False)
    pil_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    # Same detection logic as trim_whitespace in app.py
    bg = Image.new("RGB", pil_img.size, (255, 255, 255))
    diff = ImageChops.difference(pil_img, bg).convert("L")
    mask = diff.point(lambda p: 255 if p > 10 else 0)  # tolerance=10
    bbox = mask.getbbox()

    pil_img.close()

    if not bbox:
        logging.warning("Could not detect content bbox — using (0, 0) offset")
        return (0.0, 0.0)

    # bbox is (left, top, right, bottom) in detect_scale pixels
    # Apply same padding as trim_whitespace: 20 pixels at pdf_scale
    # At detect_scale, padding = 20 * detect_scale / pdf_scale
    padding_at_detect = 20 * detect_scale / pdf_scale

    left_detect = max(0, bbox[0] - padding_at_detect)
    top_detect = max(0, bbox[1] - padding_at_detect)

    # Convert from detect_scale pixels to pdf_scale pixels
    trim_left = left_detect * pdf_scale / detect_scale
    trim_top = top_detect * pdf_scale / detect_scale

    logging.info(f"  Content bbox at {detect_scale}x: left={bbox[0]}, top={bbox[1]}")
    logging.info(f"  Trim offset (at pdf_scale): left={trim_left:.1f}, top={trim_top:.1f} px")

    # Verify: post-trim dimensions should roughly match metadata
    right_detect = min(pix.width, bbox[2] + padding_at_detect)
    bottom_detect = min(pix.height, bbox[3] + padding_at_detect)
    detected_w = (right_detect - left_detect) * pdf_scale / detect_scale
    detected_h = (bottom_detect - top_detect) * pdf_scale / detect_scale
    logging.info(f"  Detected content size: {detected_w:.0f} x {detected_h:.0f} px (metadata: {img_w} x {img_h})")

    return (trim_left, trim_top)


def transform_coords(leaflet_coords: List[float], metadata: Dict[str, Any],
                     trim_offset: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """
    Transform Leaflet CRS.Simple coordinates to PyMuPDF PDF coordinates.

    Args:
        leaflet_coords: [x, y] where x=lng (positive), y=lat (negative)
        metadata: Must contain 'max_zoom' and 'quality_settings.pdf_scale'
        trim_offset: (trim_left, trim_top) in pixels at pdf_scale resolution

    Returns:
        (pdf_x, pdf_y) in PyMuPDF coordinate space (top-left origin, Y down)
    """
    x, y = leaflet_coords
    scale = 2 ** metadata["max_zoom"]
    pdf_scale = metadata["quality_settings"]["pdf_scale"]
    trim_left, trim_top = trim_offset

    # Leaflet → trimmed image pixels → pre-trim image pixels → PDF points
    pdf_x = (x * scale + trim_left) / pdf_scale
    pdf_y = (-y * scale + trim_top) / pdf_scale

    return (pdf_x, pdf_y)


def draw_polygon_on_pdf(page: fitz.Page, coordinates: List[List[List[float]]],
                       metadata: Dict[str, Any], config: Dict[str, Any],
                       overlay: str = None,
                       trim_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
    """
    Draw a filled polygon on the PDF page, optionally with centered overlay text.

    Args:
        page: PyMuPDF page object
        coordinates: GeoJSON polygon coordinates [[[x, y], [x, y], ...]]
        metadata: Metadata for coordinate transformation
        config: Styling configuration
        overlay: Optional text to display at the polygon's centroid
        trim_offset: (trim_left, trim_top) whitespace offset in pixels
    """
    # GeoJSON polygons: coordinates[0] is outer ring
    outer_ring = coordinates[0]

    logging.info(f"Drawing polygon with {len(outer_ring)} points")

    # Convert coords to PDF coordinates
    pdf_points = []
    for point in outer_ring:
        x, y = point[0], point[1]
        logging.info(f"  Leaflet: [{x}, {y}]")
        x_pdf, y_pdf = transform_coords([x, y], metadata, trim_offset)
        logging.info(f"  -> PDF: [{x_pdf:.2f}, {y_pdf:.2f}]")
        pdf_points.append(fitz.Point(x_pdf, y_pdf))

    # Draw filled polygon
    if len(pdf_points) >= 3:
        shape = page.new_shape()
        shape.draw_polyline(pdf_points)
        shape.finish(
            fill=config["fill_color"],
            color=config["stroke_color"],
            width=config["stroke_width"],
            fill_opacity=config["fill_opacity"],
            stroke_opacity=config.get("stroke_opacity", 1.0)
        )
        shape.commit()
        logging.info(f"✅ Polygon drawn!")

        # Draw overlay text at the polygon centroid if provided
        if overlay:
            # Calculate centroid of the polygon
            sum_x = sum(p.x for p in pdf_points)
            sum_y = sum(p.y for p in pdf_points)
            centroid_x = sum_x / len(pdf_points)
            centroid_y = sum_y / len(pdf_points)

            # Draw the overlay text centered at the centroid
            font_size = 10
            padding = 2
            text_width = fitz.get_text_length(overlay, fontsize=font_size)
            text_height = font_size

            # Draw white background
            bg_rect = fitz.Rect(
                centroid_x - text_width / 2 - padding,
                centroid_y - text_height / 2 - padding,
                centroid_x + text_width / 2 + padding,
                centroid_y + text_height / 2 + padding
            )
            bg_shape = page.new_shape()
            bg_shape.draw_rect(bg_rect)
            bg_shape.finish(
                fill=(1, 1, 1),  # White background
                fill_opacity=1.0  # Solid
            )
            bg_shape.commit()

            # Draw text on top
            text_x = centroid_x - text_width / 2
            text_y = centroid_y + font_size / 3

            page.insert_text(
                fitz.Point(text_x, text_y),
                overlay,
                fontsize=font_size,
                color=(0, 0, 0)  # Black text
            )
            logging.info(f"✅ Polygon overlay text drawn: '{overlay}'")
    else:
        logging.warning(f"⚠️  Not enough points: {len(pdf_points)}")


def draw_marker_on_pdf(page: fitz.Page, coordinates: List[float],
                       metadata: Dict[str, Any], config: Dict[str, Any],
                       label: str = None,
                       overlay: str = None,
                       trim_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
    """
    Draw a circular marker on the PDF page, optionally with centered overlay text.

    Args:
        page: PyMuPDF page object
        coordinates: [x, y] point coordinates
        metadata: Metadata for coordinate transformation
        config: Styling configuration
        label: Optional text label (drawn below the marker)
        overlay: Optional text centered inside the marker circle
        trim_offset: (trim_left, trim_top) whitespace offset in pixels
    """
    x, y = coordinates[0], coordinates[1]
    logging.info(f"Drawing marker at [{x}, {y}]")

    # Convert to PDF coordinates
    x_pdf, y_pdf = transform_coords([x, y], metadata, trim_offset)
    radius = config["radius"]

    # If overlay text is provided, scale the circle to fit the text
    if overlay:
        font_size = min(radius * 0.5, 8)  # Keep text readable
        # Estimate text width and ensure circle is large enough
        text_width = len(overlay) * font_size * 0.5
        min_radius = max(radius, text_width / 2 + 4, font_size / 2 + 4)
        radius = min_radius

    # Draw circle
    shape = page.new_shape()
    shape.draw_circle(fitz.Point(x_pdf, y_pdf), radius)
    shape.finish(
        fill=config["fill_color"],
        color=config.get("stroke_color", config["fill_color"]),
        width=config.get("stroke_width", 1),
        fill_opacity=config["fill_opacity"]
    )
    shape.commit()

    # Draw centered overlay text inside the circle
    if overlay:
        font_size = min(radius * 0.5, 8)  # Keep text readable
        padding = 1

        # Calculate text dimensions
        text_width = fitz.get_text_length(overlay, fontsize=font_size)
        text_height = font_size

        # Draw white background
        bg_rect = fitz.Rect(
            x_pdf - text_width / 2 - padding,
            y_pdf - text_height / 2 - padding,
            x_pdf + text_width / 2 + padding,
            y_pdf + text_height / 2 + padding
        )
        bg_shape = page.new_shape()
        bg_shape.draw_rect(bg_rect)
        bg_shape.finish(
            fill=(1, 1, 1),  # White background
            fill_opacity=1.0  # Solid
        )
        bg_shape.commit()

        # Draw text on top
        text_x = x_pdf - text_width / 2
        text_y = y_pdf + font_size / 3  # Baseline offset for visual centering
        page.insert_text(
            fitz.Point(text_x, text_y),
            overlay,
            fontsize=font_size,
            color=(0, 0, 0)  # Black text
        )
        logging.info(f"✅ Marker drawn with overlay: '{overlay}'")
    else:
        logging.info(f"✅ Marker drawn!")

    # Draw label below if provided
    if label:
        text_config = ANNOTATION_CONFIG["text"]
        draw_text_on_pdf(page, [x_pdf, y_pdf + radius + 5], label, text_config)


def draw_square_on_pdf(page: fitz.Page, coordinates: List[List[List[float]]],
                      metadata: Dict[str, Any], config: Dict[str, Any],
                      overlay: str = None,
                      trim_offset: Tuple[float, float] = (0.0, 0.0)) -> None:
    """
    Draw a filled square/rectangle on the PDF page, optionally with centered overlay text.

    Args:
        page: PyMuPDF page object
        coordinates: Rectangle coordinates (4 corner points)
        metadata: Metadata for coordinate transformation
        config: Styling configuration
        overlay: Optional text to display at the polygon's centroid
        trim_offset: (trim_left, trim_top) whitespace offset in pixels
    """
    # Treat squares the same as polygons
    draw_polygon_on_pdf(page, coordinates, metadata, config, overlay, trim_offset)


def draw_text_on_pdf(page: fitz.Page, position: List[float],
                     text: str, config: Dict[str, Any]) -> None:
    """
    Draw text with background on the PDF page.

    Args:
        page: PyMuPDF page object
        position: [x, y] position in PDF coordinates
        text: Text content to draw
        config: Text styling configuration
    """
    x, y = position
    font_size = config["font_size"]
    padding = config["padding"]

    # Estimate text width (rough approximation)
    text_width = len(text) * font_size * 0.6
    text_height = font_size

    # Draw background rectangle
    rect = fitz.Rect(
        x - padding,
        y - padding,
        x + text_width + padding,
        y + text_height + padding
    )

    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(
        fill=config["background_color"],
        fill_opacity=config["background_opacity"]
    )
    shape.commit()

    # Draw text
    page.insert_text(
        fitz.Point(x, y + font_size),  # Baseline position
        text,
        fontsize=font_size,
        color=config["font_color"]
    )


async def download_file(url: str) -> bytes:
    """
    Download a file from a URL.

    Args:
        url: File URL to download

    Returns:
        File content as bytes
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


def annotate_pdf(pdf_bytes: bytes, objects: List[Dict[str, Any]],
                metadata: Dict[str, Any]) -> bytes:
    """
    Annotate a PDF with shapes and markers.

    Args:
        pdf_bytes: Original PDF content
        objects: List of GeoJSON feature objects to draw
        metadata: Metadata containing coordinate system info

    Returns:
        Annotated PDF as bytes
    """
    # Open PDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]  # First page

    logging.info(f"=" * 80)
    logging.info(f"PDF ANNOTATION")
    logging.info(f"=" * 80)
    logging.info(f"PDF size: {page.rect.width:.2f} x {page.rect.height:.2f} points")
    logging.info(f"Image size: {metadata['source_image']['width']} x {metadata['source_image']['height']} pixels")
    logging.info(f"Objects to draw: {len(objects)}")

    # Detect whitespace trim offset (needed when PDF had margins that were cropped)
    trim_offset = detect_trim_offset(page, metadata)
    logging.info(f"Trim offset: left={trim_offset[0]:.1f}, top={trim_offset[1]:.1f} px")
    logging.info(f"=" * 80)

    # Process each object
    objects_drawn = 0
    for i, obj in enumerate(objects):
        try:
            logging.info(f"\n--- Object {i + 1}/{len(objects)} ---")
            obj_type = obj.get("properties", {}).get("type", "unknown")
            geometry = obj.get("geometry", {})
            geo_type = geometry.get("type")
            coordinates = geometry.get("coordinates", [])

            logging.info(f"Type: {obj_type}, Geometry: {geo_type}")

            if obj_type == "rectangle" or obj_type == "square" or geo_type == "Polygon":
                config = ANNOTATION_CONFIG.get("square" if obj_type == "square" else "polygon").copy()
                # Apply transparent property if present
                if obj.get("properties", {}).get("transparent") is True:
                    config["fill_opacity"] = 0
                overlay = obj.get("overlay")
                draw_polygon_on_pdf(page, coordinates, metadata, config, overlay, trim_offset)
                objects_drawn += 1

            elif obj_type == "marker" or geo_type == "Point":
                config = ANNOTATION_CONFIG["marker"].copy()
                # Apply transparent property if present
                if obj.get("properties", {}).get("transparent") is True:
                    config["fill_opacity"] = 0
                label = obj.get("properties", {}).get("content") or obj.get("properties", {}).get("label")
                overlay = obj.get("overlay")
                draw_marker_on_pdf(page, coordinates, metadata, config, label, overlay, trim_offset)
                objects_drawn += 1

            else:
                logging.warning(f"⚠️  Unknown type: {obj_type}")

        except Exception as e:
            logging.error(f"❌ Error drawing object {i + 1}: {str(e)}", exc_info=True)
            continue

    logging.info(f"\n{'=' * 80}")
    logging.info(f"COMPLETE: {objects_drawn}/{len(objects)} objects drawn")
    logging.info(f"{'=' * 80}\n")

    # Save to bytes
    output = io.BytesIO()
    doc.save(output)
    doc.close()

    return output.getvalue()


# ==========================================
# PDF ANNOTATION ENDPOINT
# ==========================================

def register_routes(app: func.FunctionApp):
    """
    Register PDF annotation routes with the function app.

    Args:
        app: Azure Function App instance
    """

    @app.route(route="pdf-annotation", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
    async def pdf_annotation(req: func.HttpRequest) -> func.HttpResponse:
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
            ]
        }

        Returns:
        {
            "success": true,
            "annotated_pdf_url": "https://...",
            "filename": "floorplan-annotation-[timestamp].pdf"
        }
        """
        try:
            # Parse request body
            try:
                body = req.get_json()
            except ValueError:
                return func.HttpResponse(
                    json.dumps({"success": False, "error": "Invalid JSON in request body"}),
                    status_code=400,
                    mimetype="application/json"
                )

            # Validate required fields
            file_url = body.get("file_url")
            metadata_url = body.get("metadata_url")
            objects = body.get("objects", [])

            if not file_url:
                return func.HttpResponse(
                    json.dumps({"success": False, "error": "file_url is required"}),
                    status_code=400,
                    mimetype="application/json"
                )

            if not metadata_url:
                return func.HttpResponse(
                    json.dumps({"success": False, "error": "metadata_url is required"}),
                    status_code=400,
                    mimetype="application/json"
                )

            # If file_url doesn't point to a PDF, derive the PDF path from metadata_url.
            # metadata_url pattern: .../floorplans/{file_id}/metadata.json
            # PDF pattern:          .../floorplans/{file_id}/{file_id}.pdf
            if not file_url.lower().endswith(".pdf"):
                base_dir = metadata_url.rsplit("/", 1)[0]  # strip "metadata.json"
                file_id = base_dir.rsplit("/", 1)[-1]       # last path segment = file_id
                derived_pdf_url = f"{base_dir}/{file_id}.pdf"
                logging.info(f"⚠️  file_url is not a PDF ({file_url}), deriving PDF URL: {derived_pdf_url}")
                file_url = derived_pdf_url

            logging.info(f"📝 Starting PDF annotation")
            logging.info(f"   PDF URL: {file_url}")
            logging.info(f"   Metadata URL: {metadata_url}")
            logging.info(f"   Objects to draw: {len(objects)}")

            # Download metadata
            logging.info("⬇️ Downloading metadata...")
            metadata_bytes = await download_file(metadata_url)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            logging.info(f"✅ Metadata loaded: {metadata.get('floorplan_id')}")

            # Download PDF
            logging.info("⬇️ Downloading PDF...")
            pdf_bytes = await download_file(file_url)
            logging.info(f"✅ PDF downloaded: {len(pdf_bytes)} bytes")

            # Annotate PDF
            logging.info("🎨 Annotating PDF...")
            annotated_pdf_bytes = annotate_pdf(pdf_bytes, objects, metadata)
            logging.info(f"✅ PDF annotated: {len(annotated_pdf_bytes)} bytes")

            # Generate filename with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

            # Extract original filename without extension
            original_filename = file_url.split("/")[-1].rsplit(".", 1)[0]
            annotated_filename = f"{original_filename}-annotation-{timestamp}.pdf"

            # Upload to Azure Blob Storage
            connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                return func.HttpResponse(
                    json.dumps({"success": False, "error": "Azure Storage connection string not configured"}),
                    status_code=500,
                    mimetype="application/json"
                )

            logging.info("☁️ Uploading to Azure Blob Storage...")
            blob_service = BlobServiceClient.from_connection_string(connection_string)

            # Upload to 'annotated-pdfs' container
            container_name = "annotated-pdfs"
            try:
                container_client = blob_service.get_container_client(container_name)
                container_client.get_container_properties()
            except:
                # Create container if it doesn't exist
                container_client = blob_service.create_container(container_name, public_access="blob")
                logging.info(f"Created container: {container_name}")

            # Upload the annotated PDF
            blob_client = blob_service.get_blob_client(container_name, annotated_filename)
            blob_client.upload_blob(
                annotated_pdf_bytes,
                overwrite=True,
                content_settings=ContentSettings(content_type="application/pdf")
            )

            # Generate the public URL
            annotated_pdf_url = f"https://blocksplayground.blob.core.windows.net/{container_name}/{annotated_filename}"

            logging.info(f"✅ Upload complete!")
            logging.info(f"   URL: {annotated_pdf_url}")

            # Return success response
            return func.HttpResponse(
                json.dumps({
                    "success": True,
                    "annotated_pdf_url": annotated_pdf_url,
                    "filename": annotated_filename,
                    "objects_drawn": len(objects),
                    "metadata": {
                        "floorplan_id": metadata.get("floorplan_id"),
                        "source_url": file_url
                    }
                }),
                status_code=200,
                mimetype="application/json"
            )

        except httpx.HTTPError as e:
            logging.error(f"❌ Download error: {str(e)}")
            return func.HttpResponse(
                json.dumps({
                    "success": False,
                    "error": f"Failed to download file: {str(e)}"
                }),
                status_code=400,
                mimetype="application/json"
            )

        except Exception as e:
            logging.error(f"❌ Error annotating PDF: {str(e)}", exc_info=True)
            return func.HttpResponse(
                json.dumps({
                    "success": False,
                    "error": f"Error annotating PDF: {str(e)}",
                    "error_type": type(e).__name__
                }),
                status_code=500,
                mimetype="application/json"
            )
