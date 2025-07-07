import os
import cv2
import numpy as np
import math
import rasterio
from rasterio.windows import Window
import time
import argparse
from pathlib import Path

# Configuration
TIFF_DIR = "./TKM_IMAGES"
OUTPUT_DIR = "./crop_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_message(message, file=None):
    """Print message to console and optionally to a file"""
    print(message)
    if file:
        ascii_message = message.replace('❌', '[ERROR]').replace('✅', '[OK]').replace('⚠️', '[WARNING]')
        file.write(ascii_message + "\n")

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

def process_coordinate(lat, lon, desc, window_size=1000, log_file=None):
    """Process a single lat/lon coordinate against all GeoTIFFs"""
    log_message(f"\n=== Processing coordinate: {lat}, {lon} ({desc}) ===", log_file)
    
    found = False
    
    for fname in os.listdir(TIFF_DIR):
        if not fname.endswith(".tif"):
            continue
            
        tif_path = os.path.join(TIFF_DIR, fname)
        log_message(f"Checking file: {fname}", log_file)
        
        try:
            with rasterio.open(tif_path) as src:
                # Log TIFF metadata
                log_message(f"  CRS: {src.crs}", log_file)
                log_message(f"  Bounds: {src.bounds}", log_file)
                
                # Determine coordinate type and transform if needed
                if abs(lat) <= 90 and abs(lon) <= 180:
                    log_message(f"  Detected WGS84 geographic coordinates: lat={lat}, lon={lon}", log_file)
                    
                    # Convert lat/lon to Web Mercator using our custom function
                    x, y = latlong_to_webmercator(lat, lon)
                    log_message(f"  Transformed to Web Mercator: x={x:.2f}, y={y:.2f}", log_file)
                else:
                    log_message(f"  Detected projected coordinates: ({lat}, {lon})", log_file)
                    
                    # Check if values need to be swapped for Web Mercator
                    if abs(lat) > abs(lon) and abs(lat) > 1000000:  # Likely swapped
                        log_message(f"  Swapping projected coordinates", log_file)
                        x, y = lat, lon
                    else:
                        x, y = lon, lat
                        
                    log_message(f"  Using projected coordinates: x={x}, y={y}", log_file)
                
                # Check if the point is within the GeoTIFF bounds
                bounds = src.bounds
                if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
                    log_message(f"  ❌ Point ({x:.2f}, {y:.2f}) outside image bounds", log_file)
                    continue
                else:
                    log_message(f"  ✅ Point ({x:.2f}, {y:.2f}) within image bounds", log_file)
                
                # Convert to pixel coordinates
                row, col = src.index(x, y)
                log_message(f"  Image pixels: row={row}, col={col}", log_file)
                
                # Calculate window parameters
                h, w = src.height, src.width
                half = window_size // 2
                
                # Calculate window offset, handling edge cases
                row_off = max(0, min(row - half, h - window_size))
                col_off = max(0, min(col - half, w - window_size))
                
                # Check if we can get a full window
                if row_off + window_size > h or col_off + window_size > w:
                    log_message(f"  ⚠️ Cannot get full {window_size}x{window_size} window, edge of image", log_file)
                    
                    # Calculate maximum possible size
                    possible_width = min(window_size, w - col_off)
                    possible_height = min(window_size, h - row_off)
                    
                    if possible_width < 100 or possible_height < 100:
                        log_message(f"  ❌ Window too small, skipping", log_file)
                        continue
                    
                    # Use partial window
                    window = Window(col_off, row_off, possible_width, possible_height)
                    log_message(f"  ⚠️ Using partial window: {window}", log_file)
                else:
                    window = Window(col_off, row_off, window_size, window_size)
                    log_message(f"  ✅ Using full window: {window}", log_file)
                
                # Read window and save image
                img = src.read([1, 2, 3], window=window)  # RGB channels
                img = np.transpose(img, (1, 2, 0)).astype(np.uint8)  # HWC
                
                # Resize to exactly 1024x1024
                img_1024 = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                log_message(f"  Resized from {img.shape} to {img_1024.shape}", log_file)
                
                # Create output filename
                out_filename = f"{desc}_{fname.split('.')[0]}_{lat}_{lon}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_filename)
                
                # Save the 1024x1024 image
                cv2.imwrite(out_path, cv2.cvtColor(img_1024, cv2.COLOR_RGB2BGR))
                log_message(f"  ✅ Saved 1024x1024 image to: {out_path}", log_file)
                
                # Create visualization with center point marked
                vis_img = img_1024.copy()
                center_y, center_x = vis_img.shape[0] // 2, vis_img.shape[1] // 2
                
                # Draw crosshair at the center (red)
                cv2.drawMarker(vis_img, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                
                # Add coordinate info
                cv2.putText(vis_img, f"lat: {lat}, lon: {lon}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Save visualization
                vis_path = os.path.join(OUTPUT_DIR, f"vis_{out_filename}")
                cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                log_message(f"  ✅ Saved visualization to: {vis_path}", log_file)
                
                found = True
                break  # Stop after first match
                
        except Exception as e:
            log_message(f"  ❌ Error processing file {fname}: {e}", log_file)
    
    return found

def main():
    parser = argparse.ArgumentParser(description='Extract GeoTIFF crop at geographic coordinates')
    parser.add_argument('lat', type=float, help='Latitude (WGS84)')
    parser.add_argument('lon', type=float, help='Longitude (WGS84)')
    parser.add_argument('--desc', type=str, default='location', help='Description for the output files')
    parser.add_argument('--size', type=int, default=1000, help='Window size in pixels')
    args = parser.parse_args()
    
    # Open log file
    log_path = os.path.join(OUTPUT_DIR, f"process_log_{int(time.time())}.txt")
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_message(f"GeoTIFF Processing - {time.ctime()}", log_file)
        log_message(f"TIFF directory: {os.path.abspath(TIFF_DIR)}", log_file)
        log_message(f"Output directory: {os.path.abspath(OUTPUT_DIR)}", log_file)
        
        # List all TIFF files
        tiff_files = [f for f in os.listdir(TIFF_DIR) if f.endswith('.tif')]
        log_message(f"Found {len(tiff_files)} TIFF files: {tiff_files}", log_file)
        
        # Process coordinate
        success = process_coordinate(args.lat, args.lon, args.desc, args.size, log_file)
        
        # Report result
        if success:
            log_message(f"✅ Successfully processed coordinate: {args.lat}, {args.lon}", log_file)
        else:
            log_message(f"❌ Failed to process coordinate: {args.lat}, {args.lon}", log_file)
        
        log_message(f"Log saved to: {log_path}")

if __name__ == "__main__":
    main()