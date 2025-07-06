import os
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform
import time
import argparse
from pathlib import Path

# Configuration
TIFF_DIR = "./TKM_Images"
OUTPUT_DIR = "./crop_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test coordinates - add your own known coordinates here
TEST_COORDINATES = [
    {"lat": 8.5, "lon": 76.9, "desc": "known_point"},
    {"lat": 8.51, "lon": 76.91, "desc": "nearby_point"},
    {"lat": 0.0, "lon": 0.0, "desc": "invalid_point"},  # Should be outside any image
    # Add more test points as needed
]

def log_message(message, file=None):
    """Print message to console and optionally to a file"""
    print(message)
    if file:
        file.write(message + "\n")

def test_coordinate(lat, lon, desc, log_file=None):
    """Test a single lat/lon coordinate against all GeoTIFFs"""
    log_message(f"\n=== Testing coordinate: {lat}, {lon} ({desc}) ===", log_file)
    
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
                log_message(f"  Shape: {src.shape}", log_file)
                
                # Step 1: Transform WGS84 (lat/lon) -> image CRS
                try:
                    dst_crs = src.crs
                    x, y = transform("EPSG:4326", dst_crs, [lon], [lat])
                    x, y = x[0], y[0]
                    log_message(f"  Transformed coordinates: {x}, {y}", log_file)
                except Exception as e:
                    log_message(f"  ❌ Coordinate transformation failed: {e}", log_file)
                    continue
                
                # Step 2: Check if point is within image bounds
                try:
                    row, col = src.index(x, y)
                    log_message(f"  Image pixels: row={row}, col={col}", log_file)
                except Exception as e:
                    log_message(f"  ❌ Point outside image bounds: {e}", log_file)
                    continue
                
                # Step 3: Calculate window parameters
                h, w = src.height, src.width
                size = 1024
                half = size // 2
                
                # Calculate window offset, handling edge cases
                row_off = max(0, min(row - half, h - size))
                col_off = max(0, min(col - half, w - size))
                
                # Check if we can get a full window
                if row_off + size > h or col_off + size > w:
                    log_message(f"  ⚠️ Cannot get full 1024x1024 window, edge of image", log_file)
                    log_message(f"  Image size: {w}x{h}, Requested window: {col_off}:{col_off+size}, {row_off}:{row_off+size}", log_file)
                    
                    # Calculate maximum possible size
                    possible_width = min(size, w - col_off)
                    possible_height = min(size, h - row_off)
                    log_message(f"  Maximum possible window size: {possible_width}x{possible_height}", log_file)
                    
                    if possible_width < 100 or possible_height < 100:
                        log_message(f"  ❌ Window too small, skipping", log_file)
                        continue
                    
                    # For testing purposes, continue with partial window
                    window = Window(col_off, row_off, possible_width, possible_height)
                    log_message(f"  ⚠️ Using partial window: {window}", log_file)
                else:
                    window = Window(col_off, row_off, size, size)
                    log_message(f"  ✅ Using full window: {window}", log_file)
                
                # Step 4: Read window and save image
                img = src.read([1, 2, 3], window=window)  # RGB channels
                img = np.transpose(img, (1, 2, 0)).astype(np.uint8)  # HWC
                
                log_message(f"  Image array shape: {img.shape}", log_file)
                log_message(f"  Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.2f}", log_file)
                
                # Create output filename
                out_filename = f"{desc}_from_{fname.split('.')[0]}_{lat}_{lon}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_filename)
                
                # Save the image
                cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                log_message(f"  ✅ Saved image to: {out_path}", log_file)
                
                # Create a visualization with the center point marked
                vis_img = img.copy()
                center_y, center_x = row - row_off, col - col_off
                
                # Check if center is within the cropped image
                if 0 <= center_x < vis_img.shape[1] and 0 <= center_y < vis_img.shape[0]:
                    # Draw crosshair at the target point (red)
                    cv2.drawMarker(vis_img, (center_x, center_y), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
                    
                    # Add coordinate text
                    cv2.putText(vis_img, f"lat: {lat}, lon: {lon}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Save visualization
                    vis_path = os.path.join(OUTPUT_DIR, f"vis_{out_filename}")
                    cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                    log_message(f"  ✅ Saved visualization to: {vis_path}", log_file)
                else:
                    log_message(f"  ⚠️ Target point ({center_x}, {center_y}) outside cropped region", log_file)
                
                # Coordinate found in this file
                found = True
                
                # Test back-transformation (pixel to geo)
                # For a point we know is in the image (the center)
                px_center_x, px_center_y = vis_img.shape[1] // 2, vis_img.shape[0] // 2
                
                # Convert to source pixel coordinates
                src_px_x = col_off + px_center_x
                src_px_y = row_off + px_center_y
                
                # Get CRS coordinates
                geo_x, geo_y = src.xy(src_px_y, src_px_x)
                
                # Convert back to lat/lon
                geo_lon, geo_lat = transform(dst_crs, "EPSG:4326", [geo_x], [geo_y])
                log_message(f"  Pixel ({px_center_x}, {px_center_y}) -> geo: ({geo_lon[0]}, {geo_lat[0]})", log_file)
                
        except Exception as e:
            log_message(f"  ❌ Error processing file {fname}: {e}", log_file)
    
    if not found:
        log_message(f"❌ Coordinate {lat}, {lon} ({desc}) not found in any image", log_file)
    else:
        log_message(f"✅ Coordinate {lat}, {lon} ({desc}) successfully processed", log_file)
    
    return found

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test GeoTIFF access and cropping')
    parser.add_argument('--add-coord', nargs=3, metavar=('LAT', 'LON', 'DESC'), 
                        help='Add a test coordinate: latitude longitude description')
    args = parser.parse_args()
    
    # Add any command line coordinates to the test set
    if args.add_coord:
        lat, lon, desc = args.add_coord
        TEST_COORDINATES.append({"lat": float(lat), "lon": float(lon), "desc": desc})
    
    # Open log file
    log_path = os.path.join(OUTPUT_DIR, f"tiff_test_log_{int(time.time())}.txt")
    with open(log_path, 'w') as log_file:
        log_message(f"GeoTIFF Access Test - {time.ctime()}", log_file)
        log_message(f"TIFF directory: {os.path.abspath(TIFF_DIR)}", log_file)
        log_message(f"Output directory: {os.path.abspath(OUTPUT_DIR)}", log_file)
        
        # List all TIFF files
        tiff_files = [f for f in os.listdir(TIFF_DIR) if f.endswith('.tif')]
        log_message(f"Found {len(tiff_files)} TIFF files: {tiff_files}", log_file)
        
        # Process each test coordinate
        results = []
        for coord in TEST_COORDINATES:
            result = test_coordinate(coord["lat"], coord["lon"], coord["desc"], log_file)
            results.append((coord, result))
        
        # Summary
        log_message("\n=== SUMMARY ===", log_file)
        for (coord, result) in results:
            status = "✅ Found" if result else "❌ Not found"
            log_message(f"{status}: {coord['lat']}, {coord['lon']} ({coord['desc']})", log_file)
        
        log_message(f"\nLog saved to: {log_path}")

if __name__ == "__main__":
    main()