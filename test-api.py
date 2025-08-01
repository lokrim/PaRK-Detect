from flask import Flask, request, jsonify
from flask_cors import CORS
import os, cv2, torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform
from networks.dinknet import DinkNet34_WithBranch
from torch.autograd import Variable as V
import time
import json
import math
import sys
import datetime

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS for all routes

# Path to directory containing all Panchayat-level GeoTIFFs
TIFF_DIR = "./TKM_Images"
# Folder to save cropped debug images
CROPPED_IMG_DIR = "./cropped_debug"
DEBUG_DIR = "./debug_outputs"
os.makedirs(CROPPED_IMG_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

class TTAFrame:
    def __init__(self, net, weights_path):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=list(range(torch.cuda.device_count())))
        self.net.load_state_dict(torch.load(weights_path))
        self.net.eval()

    def preprocess(self, img):
        # Match exact preprocessing from test_without_TTA
        arr = np.expand_dims(np.array(img), axis=0).transpose(0,3,1,2).astype(np.float32)
        arr = arr/255.0*3.2 - 1.6
        return V(torch.Tensor(arr).cuda())
    
    def infer(self, img):
        """Process image through model and return outputs - identical to test_without_TTA"""
        with torch.no_grad():
            img_tensor = self.preprocess(img)
            outputs = self.net.forward(img_tensor)
            
            # Match exact output format from test_without_TTA
            mask = outputs[0].squeeze().cpu().data.numpy()
            prob = outputs[1].squeeze(0).cpu().data.numpy()  # Keep [0,i,j] dimension
            posi = outputs[2].squeeze().cpu().data.numpy()
            link = outputs[3].squeeze().cpu().data.numpy()
            
            return mask, prob, posi, link

    def postprocess_full(self, mask, prob, posi, link):
        """Complete post-processing exactly matching test_without_TTA"""
        # Match exact thresholding logic from test_without_TTA
        mask[mask > 0.1] = 255
        mask[mask <= 0.5] = 0
        prob[prob > 0.1] = 1
        prob[prob <= 0.5] = 0
        link[link > 0.1] = 1
        link[link <= 0.5] = 0

        # Decode keypoint coordinates from normalized 16x16 cell-based predictions
        posi_final = np.zeros((2, 64, 64), np.int64)
        for i in range(64):
            for j in range(64):
                if prob[0, i, j] == 1:
                    posi_final[0, i, j] = int(posi[0, i, j] * 15 + 0.5) + i * 16
                    posi_final[1, i, j] = int(posi[1, i, j] * 15 + 0.5) + j * 16

                    # Remove invalid links to non-existent keypoints
                    for d, (di, dj) in enumerate([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]):
                        ni, nj = i + di, j + dj
                        if not (0 <= ni < 64 and 0 <= nj < 64) or prob[0, ni, nj] != 1:
                            link[d, i, j] = 0

                    # Add diagonal links if neighbor exists and straight neighbors are empty
                    if i - 1 >= 0 and j + 1 < 64 and link[1, i, j] == 0:
                        if prob[0, i - 1, j + 1] == 1 and prob[0, i - 1, j] == 0 and prob[0, i, j + 1] == 0:
                            link[1, i, j] = 1
                    if i + 1 < 64 and j + 1 < 64 and link[3, i, j] == 0:
                        if prob[0, i + 1, j + 1] == 1 and prob[0, i + 1, j] == 0 and prob[0, i, j + 1] == 0:
                            link[3, i, j] = 1
                    if i + 1 < 64 and j - 1 >= 0 and link[5, i, j] == 0:
                        if prob[0, i + 1, j - 1] == 1 and prob[0, i + 1, j] == 0 and prob[0, i, j - 1] == 0:
                            link[5, i, j] = 1
                    if i - 1 >= 0 and j - 1 >= 0 and link[7, i, j] == 0:
                        if prob[0, i - 1, j - 1] == 1 and prob[0, i - 1, j] == 0 and prob[0, i, j - 1] == 0:
                            link[7, i, j] = 1
                else:
                    posi_final[0, i, j] = -1
                    posi_final[1, i, j] = -1
                    link[:, i, j] = -1

        # Remove small cycles - exactly as in test_without_TTA
        posi_cal = posi_final.astype(np.int64)
        for i in range(63):
            for j in range(63):
                if all(prob[0, ni, nj] == 1 for ni, nj in [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]):
                    a = np.sum((posi_cal[:, i, j] - posi_cal[:, i + 1, j]) ** 2)
                    b = np.sum((posi_cal[:, i + 1, j] - posi_cal[:, i + 1, j + 1]) ** 2)
                    c = np.sum((posi_cal[:, i, j] - posi_cal[:, i, j + 1]) ** 2)
                    d = np.sum((posi_cal[:, i, j + 1] - posi_cal[:, i + 1, j + 1]) ** 2)
                    max_dist = max(a, b, c, d)
                    if a == max_dist:
                        link[4, i, j] = link[0, i + 1, j] = 0
                    elif b == max_dist:
                        link[2, i + 1, j] = link[6, i + 1, j + 1] = 0
                    elif c == max_dist:
                        link[2, i, j] = link[6, i, j + 1] = 0
                    elif d == max_dist:
                        link[4, i, j + 1] = link[0, i + 1, j + 1] = 0
        
        return posi_final, link

    def generate_debug_visualizations(self, img, mask, prob, posi_final, link, debug_prefix):
        """Generate all debug visualizations exactly like test_without_TTA"""
        debug_paths = {}
        
        # 1. Save mask as RGB image
        mask_rgb = np.concatenate([mask[:, :, None]] * 3, axis=2)
        mask_path = os.path.join(DEBUG_DIR, f"{debug_prefix}_mask.png")
        cv2.imwrite(mask_path, mask_rgb.astype(np.uint8))
        debug_paths['mask'] = mask_path
        
        # 2. Generate overlay visualization of anchors and links (prob_posi_link)
        new_img = np.zeros((1024, 1024, 3), np.uint8)
        for i in range(64):
            for j in range(64):
                if prob[0, i, j] == 0:
                    new_img[i*16:(i+1)*16, j*16:(j+1)*16] = [0, 255, 255]  # Yellow for non-keypoints
                else:
                    new_img[i*16:(i+1)*16, j*16:(j+1)*16] = [255, 255, 255]  # White for keypoints
                    # Draw links
                    for d, (di, dj) in enumerate([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 64 and 0 <= nj < 64 and link[d, i, j] == 1 and prob[0, ni, nj] == 1:
                            pt1 = (posi_final[1, i, j], posi_final[0, i, j])
                            pt2 = (posi_final[1, ni, nj], posi_final[0, ni, nj])
                            cv2.line(new_img, pt1, pt2, (0, 255, 0), 1)  # Green links
        
        # Draw keypoints as red dots
        for i in range(64):
            for j in range(64):
                if prob[0, i, j] == 1:
                    m, n = posi_final[0, i, j], posi_final[1, i, j]
                    if 0 <= m < 1024 and 0 <= n < 1024:
                        new_img[m, n] = [0, 0, 255]  # Red keypoints
        
        # Save keypoint+link visualization
        keypoint_path = os.path.join(DEBUG_DIR, f"{debug_prefix}_prob_posi_link.png")
        cv2.imwrite(keypoint_path, new_img)
        debug_paths['keypoints_links'] = keypoint_path
        
        # 3. Merge satellite image with link visualization
        sat_merge = cv2.addWeighted(img, 0.8, new_img, 0.2, 0)
        merge_path = os.path.join(DEBUG_DIR, f"{debug_prefix}_merge.png")
        cv2.imwrite(merge_path, cv2.cvtColor(sat_merge, cv2.COLOR_RGB2BGR))
        debug_paths['merge'] = merge_path
        
        return debug_paths

    # Add this function to the TTAFrame class
    def save_raw_model_outputs(self, mask, prob, posi, link, debug_prefix):
        """Save raw model outputs to files for debugging"""
        output_paths = {}
        
        # Save mask as numpy array
        mask_path = os.path.join(DEBUG_DIR, f"{debug_prefix}_raw_mask.npy")
        np.save(mask_path, mask)
        output_paths['raw_mask_npy'] = mask_path
        
        # Save probability map as numpy array
        prob_path = os.path.join(DEBUG_DIR, f"{debug_prefix}_raw_prob.npy")
        np.save(prob_path, prob)
        output_paths['raw_prob_npy'] = prob_path
        
        # Save position map as numpy array
        posi_path = os.path.join(DEBUG_DIR, f"{debug_prefix}_raw_posi.npy")
        np.save(posi_path, posi)
        output_paths['raw_posi_npy'] = posi_path
        
        # Save link map as numpy array
        link_path = os.path.join(DEBUG_DIR, f"{debug_prefix}_raw_link.npy")
        np.save(link_path, link)
        output_paths['raw_link_npy'] = link_path
        
        return output_paths

# Initialize model
model = TTAFrame(DinkNet34_WithBranch, "weights/alan_weight.th")

# Add the latlong_to_webmercator function from test-geoTIFF.py
def latlong_to_webmercator(lat, lon):
    """
    Convert latitude/longitude to Web Mercator (EPSG:3857) coordinates
    This is a direct implementation to avoid PROJ database issues
    """
    # Constants for the Earth's radius in meters
    EARTH_RADIUS = 6378137.0
    
    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # Calculate the x coordinate
    x = EARTH_RADIUS * lon_rad
    
    # Calculate the y coordinate
    y = EARTH_RADIUS * math.log(math.tan(math.pi/4 + lat_rad/2))
    
    return x, y

def webmercator_to_wgs84(x, y):
    """
    Convert Web Mercator coordinates (EPSG:3857) to WGS84 (EPSG:4326)
    This is a direct implementation to avoid PROJ database issues
    """
    # Constants for the Earth's radius in meters
    EARTH_RADIUS = 6378137.0
    
    # Convert to longitude and latitude
    lon = x * 180.0 / (EARTH_RADIUS * math.pi)
    lat = math.atan(math.exp(y / EARTH_RADIUS)) * 2.0 - math.pi/2.0
    lat = lat * 180.0 / math.pi
    
    return lon, lat

def ensure_wgs84_format(geo_x, geo_y):
    """
    Ensure coordinates are in WGS84 format (longitude, latitude)
    and within the valid range: lon [-180,180], lat [-90,90]
    """
    # Check if values appear to be Web Mercator coordinates
    if abs(geo_x) > 180 or abs(geo_y) > 90:
        # Convert from Web Mercator to WGS84
        lon, lat = webmercator_to_wgs84(geo_x, geo_y)
        return lon, lat
    else:
        # Already in WGS84 format, just ensure longitude is first
        return geo_x, geo_y

def find_geotiff_for_coordinates(lat, lon):
    """Find which GeoTIFF contains the given coordinates"""
    found_tiffs = []
    
    for fname in os.listdir(TIFF_DIR):
        if not fname.endswith(".tif"):
            continue
            
        tif_path = os.path.join(TIFF_DIR, fname)
        try:
            with rasterio.open(tif_path) as src:
                # Determine coordinate type and transform if needed
                if abs(lat) <= 90 and abs(lon) <= 180:
                    # Convert lat/lon to Web Mercator using our custom function
                    x, y = latlong_to_webmercator(lat, lon)
                else:
                    # Check if values need to be swapped for Web Mercator
                    if abs(lat) > abs(lon) and abs(lat) > 1000000:  # Likely swapped
                        x, y = lat, lon
                    else:
                        x, y = lon, lat
                
                # Check if the point is within the GeoTIFF bounds
                bounds = src.bounds
                if (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
                    found_tiffs.append({
                        'filename': fname,
                        'bounds': {
                            'left': bounds.left,
                            'bottom': bounds.bottom,
                            'right': bounds.right,
                            'top': bounds.top
                        },
                        'crs': str(src.crs),
                        'shape': src.shape,
                        'path': os.path.join(TIFF_DIR, fname)
                    })
        except Exception as e:
            continue
    
    return found_tiffs

def extract_geotiff_window(tiff_path, lat, lon, window_size=1000):
    """Extract a window from GeoTIFF at specified coordinates and return a 1024x1024 image"""
    try:
        with rasterio.open(tiff_path) as src:
            # Determine coordinate type and transform if needed
            if abs(lat) <= 90 and abs(lon) <= 180:
                # Convert lat/lon to Web Mercator using our custom function
                x, y = latlong_to_webmercator(lat, lon)
            else:
                # Check if values need to be swapped for Web Mercator
                if abs(lat) > abs(lon) and abs(lat) > 1000000:  # Likely swapped
                    x, y = lat, lon
                else:
                    x, y = lon, lat
            
            # Check if the point is within the GeoTIFF bounds
            bounds = src.bounds
            if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
                return None, "Coordinates outside image bounds"
            
            # Convert to pixel coordinates
            row, col = src.index(x, y)
            
            # Calculate window parameters
            h, w = src.height, src.width
            half = window_size // 2
            
            # Calculate window offset, handling edge cases
            row_off = max(0, min(row - half, h - window_size))
            col_off = max(0, min(col - half, w - window_size))
            
            # Check if we can get a full window
            if row_off + window_size > h or col_off + window_size > w:
                # Calculate maximum possible size
                possible_width = min(window_size, w - col_off)
                possible_height = min(window_size, h - row_off)
                
                if possible_width < 512 or possible_height < 512:
                    return None, "Window too small for quality results"
                
                # Use partial window
                window = Window(col_off, row_off, possible_width, possible_height)
            else:
                window = Window(col_off, row_off, window_size, window_size)
            
            # Read window and process image
            img = src.read([1, 2, 3], window=window)  # RGB channels
            img = np.transpose(img, (1, 2, 0)).astype(np.uint8)  # HWC
            
            # Resize to exactly 1024x1024 for model
            img_1024 = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            
            return {
                'image': img_1024,
                'window': {
                    'row_off': row_off,
                    'col_off': col_off,
                    'width': window.width,
                    'height': window.height
                },
                'pixel_coords': {
                    'row': row,
                    'col': col
                },
                'src_crs': src.crs,
                'src_transform': src.transform,
                'src_bounds': src.bounds
            }, None
            
    except Exception as e:
        return None, str(e)

@app.route("/find_tiff", methods=["GET"])
def find_tiff():
    """Endpoint to find which GeoTIFF contains given coordinates"""
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    if lat is None or lon is None:
        return jsonify({"error": "Missing lat or lon"}), 400
    
    found_tiffs = find_geotiff_for_coordinates(lat, lon)
    
    if not found_tiffs:
        return jsonify({
            "coordinate": {"lat": lat, "lon": lon},
            "found_tiffs": [],
            "message": "No GeoTIFF found containing these coordinates"
        }), 404
    
    return jsonify({
        "coordinate": {"lat": lat, "lon": lon},
        "found_tiffs": found_tiffs,
        "message": f"Found {len(found_tiffs)} GeoTIFF(s) containing these coordinates"
    })

def save_geojson(data, lat, lon):
    """Save GeoJSON data to a file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"road_detection_{lat}_{lon}_{timestamp}.geojson"
    
    # Create output directory for GeoJSON files
    geojson_dir = "./geojson_output"
    os.makedirs(geojson_dir, exist_ok=True)
    
    filepath = os.path.join(geojson_dir, filename)
    
    # Save the GeoJSON with nice formatting
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath

@app.route("/infer_coord", methods=["GET"])
def infer_coord():
    start_time = time.time()
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    debug = request.args.get("debug", type=bool, default=True)  # Enable debug by default
    
    if lat is None or lon is None:
        return jsonify({"error": "Missing lat or lon"}), 400

    print(f"Processing coordinates: lat={lat}, lon={lon}")

    # Debug log
    debug_info = {
        "input_coordinates": {"lat": lat, "lon": lon},
        "processing_steps": [],
        "tiff_search_results": []
    }

    # First, find all GeoTIFFs containing these coordinates using the improved function
    found_tiffs = find_geotiff_for_coordinates(lat, lon)
    debug_info["tiff_search_results"] = found_tiffs
    
    if not found_tiffs:
        return jsonify({
            "error": "Coordinate not found in any GeoTIFF",
            "debug_info": debug_info
        }), 404

    # Process the first suitable GeoTIFF
    for tiff_info in found_tiffs:
        fname = tiff_info['filename']
        tif_path = tiff_info['path']
        
        debug_info["processing_steps"].append(f"Processing {fname}")

        try:
            # Extract window using the improved extraction function
            extraction_result, error = extract_geotiff_window(tif_path, lat, lon, window_size=1000)
            
            if error or extraction_result is None:
                debug_info["processing_steps"].append(f"Error extracting window: {error}")
                continue
                
            img = extraction_result['image']
            window_info = extraction_result['window']
            pixel_coords = extraction_result['pixel_coords']
            src_crs = extraction_result['src_crs']
            
            debug_info["processing_steps"].append(f"Window size: {window_info['width']}x{window_info['height']}")
            debug_info["processing_steps"].append(f"Pixel coordinates: row={pixel_coords['row']}, col={pixel_coords['col']}")
            debug_info["processing_steps"].append(f"Resized to 1024x1024 for model input")

            # Save input image for debugging
            debug_prefix = f"{lat}_{lon}_{int(time.time())}"
            input_path = os.path.join(CROPPED_IMG_DIR, f"input_{debug_prefix}.jpg")
            cv2.imwrite(input_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            debug_info["input_image"] = input_path

            # Run model inference
            debug_info["processing_steps"].append("Running model inference")
            mask, prob, posi, link = model.infer(img)
            
            # Save raw model outputs if debug is enabled
            if debug:
                raw_output_paths = model.save_raw_model_outputs(mask, prob, posi, link, debug_prefix)
                debug_info["raw_model_outputs"] = raw_output_paths

            # Apply post-processing
            debug_info["processing_steps"].append("Applying post-processing")
            posi_final, link_processed = model.postprocess_full(mask, prob, posi, link)
            
            # Count detected keypoints
            keypoint_count = np.sum(prob[0, :, :] == 1)
            debug_info["keypoint_count"] = int(keypoint_count)
            debug_info["processing_steps"].append(f"Detected {keypoint_count} keypoints")

            # Generate debug visualizations
            if debug:
                debug_paths = model.generate_debug_visualizations(
                    img, mask, prob, posi_final, link_processed, debug_prefix
                )
                debug_info["debug_images"] = debug_paths

            # Create GeoJSON features
            features = []
            dirs = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
            
            def pixel_to_geo(px, py):
                """Convert pixel coordinates to geographic coordinates"""
                # Print progress indicator to console
                sys.stdout.write(".")
                sys.stdout.flush()
                
                # Scale back to original window size
                scale_factor = window_info['width'] / 1024  # Assuming square window
                actual_px = px * scale_factor
                actual_py = py * scale_factor
                
                # Convert to source coordinates
                src_px = window_info['col_off'] + actual_px
                src_py = window_info['row_off'] + actual_py
                
                # Convert to geographic coordinates
                with rasterio.open(tif_path) as src:
                    geo_x, geo_y = src.xy(src_py, src_px)
                    
                    # Convert to WGS84 if needed
                    if src_crs.to_string() == 'EPSG:4326' or 'WGS 84' in str(src_crs):
                        # Directly return as longitude, latitude for GeoJSON
                        return ensure_wgs84_format(geo_x, geo_y)
                    else:
                        # Try direct transformation using rasterio's transform
                        try:
                            lon_out, lat_out = transform(src_crs, "EPSG:4326", [geo_x], [geo_y])
                            # Return as longitude, latitude for GeoJSON
                            return ensure_wgs84_format(lon_out[0], lat_out[0])
                        except Exception:
                            # If that fails, use our direct conversion function
                            lon, lat = webmercator_to_wgs84(geo_x, geo_y)
                            return lon, lat  # Already in longitude, latitude order
            
            # Generate GeoJSON features with MultiLineString format
            road_segments = {}  # Dictionary to collect line segments
            
            for i in range(64):
                for j in range(64):
                    if prob[0, i, j] == 1:
                        x1 = posi_final[1, i, j]
                        y1 = posi_final[0, i, j]
                        
                        if x1 < 0 or y1 < 0:
                            continue
                            
                        geo_x1, geo_y1 = pixel_to_geo(x1, y1)
                        
                        # Process each direction
                        for d, (di, dj) in enumerate(dirs):
                            ni, nj = i+di, j+dj
                            if 0 <= ni < 64 and 0 <= nj < 64 and link_processed[d, i, j] == 1:
                                x2 = posi_final[1, ni, nj]
                                y2 = posi_final[0, ni, nj]
                                
                                if x2 < 0 or y2 < 0:
                                    continue
                                    
                                geo_x2, geo_y2 = pixel_to_geo(x2, y2)
                                
                                # Create a unique road ID (using i,j as road identifier)
                                road_id = f"road_{i}_{j}"
                                
                                # Add segment to road collection
                                if road_id not in road_segments:
                                    road_segments[road_id] = []
                                
                                # Add the coordinates with z-value (0.0) for compatibility
                                road_segments[road_id].append([
                                    [geo_x1, geo_y1, 0.0],
                                    [geo_x2, geo_y2, 0.0]
                                ])
            
            # Convert segments to MultiLineString features
            features = []
            for idx, (road_id, segments) in enumerate(road_segments.items()):
                features.append({
                    "type": "Feature",
                    "properties": {
                        "OBJECTID_1": idx + 1,
                        "id": None,
                        "lsgdcode": "G020303",
                        "munci": None,
                        "panch": "Melila",
                        "block": "Vettikkavala",
                        "localbody": "Grama Panchayat",
                        "roadname": f"DETECTED_ROAD_{idx+1}",
                        "roadid": f"AUTO-{idx+1:06d}",
                        "district": "Kollam",
                        "roadcode": None,
                        "category": "VR",
                        "surfacetyp": "Concrete",
                        "roadtype": "AWR",
                        "width": "3",
                        "carriagewa": "3",
                        "formationw": "3",
                        "soiltype": "Laterite",
                        "terraintyp": "Plain",
                        "roadlength": 0.0,  # Could calculate actual length
                        "year_maint": None,
                        "junction_n": None,
                        "junction_l": "/",
                        "junction_r": None,
                        "junction_j": None,
                        "road_start": "-",
                        "road_locn_": f"{lat}/{lon}",
                        "road_remar": ":/:/:/:",
                        "layer": None,
                        "path": None,
                        "StartChain": 0.0,
                        "EndChain": 0.0,
                        "Grid": 0.0,
                        "external_s": "0",
                        "Corporatio": None,
                        "ownership": None,
                        "OBJECTID": 0,
                        "Shape_Leng": 0.0,
                        "Shape_Le_1": 0.0
                    },
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": segments
                    }
                })
            
            debug_info["feature_count"] = len(features)
            debug_info["link_count"] = sum(len(segments) for segments in road_segments.values())
            
            process_time = time.time() - start_time
            debug_info["processing_steps"].append(f"Total processing time: {process_time:.2f}s")
            
            # Return complete response
            response = {
                "type": "FeatureCollection",
                "name": f"Road_Detection_{lat}_{lon}",
                "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
                "features": features,
                "metadata": {
                    "source_tiff": fname,
                    "input_coordinates": {"lat": lat, "lon": lon},
                    "coverage_meters": 500,
                    "gsd_meters": 0.5,
                    "keypoint_count": int(keypoint_count),
                    "feature_count": len(features),
                    "process_time_seconds": process_time
                }
            }
            
            # Save GeoJSON to file
            print("\nSaving GeoJSON to file...")
            geojson_path = save_geojson(response, lat, lon)
            debug_info["geojson_file"] = geojson_path
            print(f"GeoJSON saved to: {geojson_path}")

            if debug:
                response["debug_info"] = debug_info

            return jsonify(response)
                
        except Exception as e:
            debug_info["processing_steps"].append(f"Error processing {fname}: {str(e)}")
            continue

    return jsonify({
        "error": "No suitable GeoTIFF could be processed",
        "debug_info": debug_info
    }), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("Server running on port 5000")


