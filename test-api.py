from flask import Flask, request, jsonify
import os, cv2, torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform
from networks.dinknet import DinkNet34_WithBranch
from torch.autograd import Variable as V
import time

app = Flask(__name__)

# Path to directory containing all Panchayat-level GeoTIFFs
TIFF_DIR = "./TKM_Images"
# Folder to save cropped debug images
CROPPED_IMG_DIR = "./cropped_debug"
os.makedirs(CROPPED_IMG_DIR, exist_ok=True)

class TTAFrame:
    def __init__(self, net, weights_path):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=list(range(torch.cuda.device_count())))
        self.net.load_state_dict(torch.load(weights_path))
        self.net.eval()

    def preprocess(self, img):
        arr = np.expand_dims(np.array(img), axis=0).transpose(0,3,1,2).astype(np.float32)
        arr = arr/255.0*3.2 - 1.6
        return V(torch.Tensor(arr).cuda())
    
    def infer(self, img):
        """Process image through model and return outputs - identical to test_without_TTA"""
        with torch.no_grad():
            img_tensor = self.preprocess(img)
            outputs = self.net.forward(img_tensor)
            
            # Match the exact return format from test_without_TTA_mask_keypoints_link.py
            mask = outputs[0].squeeze().cpu().data.numpy()
            prob = outputs[1].squeeze(0).cpu().data.numpy()  # Keep [0,i,j] dimension
            posi = outputs[2].squeeze().cpu().data.numpy()
            link = outputs[3].squeeze().cpu().data.numpy()
            
            return mask, prob, posi, link

    def postprocess_full(self, mask, prob, posi, link):
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
                if prob[0, i, j] == 1:  # Now accessing with [0,i,j]
                    posi_final[0, i, j] = int(posi[0, i, j] * 15 + 0.5) + i * 16
                    posi_final[1, i, j] = int(posi[1, i, j] * 15 + 0.5) + j * 16

                    # Remove invalid links to non-existent keypoints
                    for d, (di, dj) in enumerate([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]):
                        ni, nj = i + di, j + dj
                        if not (0 <= ni < 64 and 0 <= nj < 64) or prob[0, ni, nj] != 1:  # Updated to prob[0,ni,nj]
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
                    link[:, i, j] = -1  # Added this line to match test_without_TTA

        # Remove small cycles using the same approach as test_without_TTA
        self.RemoveCircle_Update(posi_final, prob, link)
        
        return posi_final, link
    
    def RemoveCircle_Update(self, posi_final, prob, link):
        """Removes small 2x2 cycles in road network, exactly as in test_without_TTA"""
        posi_cal = posi_final.astype(np.int64)  # Match original implementation
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

# Initialize model
model = TTAFrame(DinkNet34_WithBranch, "weights/log04_dink34.th")

@app.route("/infer_coord", methods=["GET"])
def infer_coord():
    start_time = time.time()  # Track processing time
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    if lat is None or lon is None:
        return jsonify({"error": "Missing lat or lon"}), 400

    # Loop through all Panchayat-level GeoTIFFs
    for fname in os.listdir(TIFF_DIR):
        if not fname.endswith(".tif"):
            continue

        tif_path = os.path.join(TIFF_DIR, fname)

        with rasterio.open(tif_path) as src:
            # Transform WGS84 (lat/lon) -> image CRS
            try:
                dst_crs = src.crs
                x, y = transform("EPSG:4326", dst_crs, [lon], [lat])
                x, y = x[0], y[0]
            except Exception as e:
                continue

            # Convert map coords to pixel coords
            try:
                row, col = src.index(x, y)
            except Exception:
                continue  # coordinate not inside this TIFF

            h, w = src.height, src.width
            # Define a 1024×1024 pixel window centered at (col, row)
            size = 1024
            half = size // 2
            
            # Handle edge cases by adjusting the window
            row_off = max(0, min(row - half, h - size))
            col_off = max(0, min(col - half, w - size))
            
            # Skip if we can't get a full 1024x1024 window
            if row_off + size > h or col_off + size > w:
                continue

            window = Window(col_off, row_off, size, size)
            img = src.read([1, 2, 3], window=window)  # RGB channels
            img = np.transpose(img, (1, 2, 0)).astype(np.uint8)  # HWC

            # Save debug image to disk
            debug_path = os.path.join(CROPPED_IMG_DIR, f"crop_{lat}_{lon}.jpg")
            cv2.imwrite(debug_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Run inference on cropped patch - use exact pipeline from test_without_TTA
            mask, prob, posi, link = model.infer(img)
            
            # Apply post-processing (same as test_without_TTA_mask_keypoints_link.py)
            posi_final, link_processed = model.postprocess_full(mask, prob, posi, link)

            # Decode keypoints to GeoJSON with georeferencing
            features = []
            dirs = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
            
            # Function to convert pixel coordinates to geographic coordinates
            def pixel_to_geo(px, py):
                # Convert from window pixel coords to src pixel coords
                src_px = col_off + px
                src_py = row_off + py
                
                # Convert from pixel coords to CRS coords
                geo_x, geo_y = src.xy(src_py, src_px)
                
                # Convert from image CRS to WGS84
                lon, lat = transform(dst_crs, "EPSG:4326", [geo_x], [geo_y])
                return lon[0], lat[0]
            
            # Create GeoJSON features
            for i in range(64):
                for j in range(64):
                    if prob[0, i, j] == 1:  # Updated to access prob[0,i,j]
                        x1 = posi_final[1, i, j]
                        y1 = posi_final[0, i, j]
                        
                        if x1 < 0 or y1 < 0:  # Skip invalid points
                            continue
                            
                        # Convert to geographic coordinates
                        geo_x1, geo_y1 = pixel_to_geo(x1, y1)
                        
                        for d, (di, dj) in enumerate(dirs):
                            ni, nj = i+di, j+dj
                            if 0 <= ni < 64 and 0 <= nj < 64 and link_processed[d, i, j] == 1:
                                x2 = posi_final[1, ni, nj]
                                y2 = posi_final[0, ni, nj]
                                
                                if x2 < 0 or y2 < 0:  # Skip invalid points
                                    continue
                                    
                                # Convert to geographic coordinates
                                geo_x2, geo_y2 = pixel_to_geo(x2, y2)
                                
                                features.append({
                                    "type": "Feature",
                                    "geometry": {
                                        "type": "LineString", 
                                        "coordinates": [[geo_x1, geo_y1], [geo_x2, geo_y2]]
                                    },
                                    "properties": {"cell": [i, j], "dir": d}
                                })
            
            process_time = time.time() - start_time
            return jsonify({
                "type": "FeatureCollection", 
                "features": features, 
                "debug_image": debug_path,
                "source_tiff": fname,
                "process_time_seconds": process_time
            })

    return jsonify({"error": "Coordinate not found in any image"}), 404


#curl "http://localhost:5000/infer_coord?lat=8.5&lon=76.9" for testing
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


