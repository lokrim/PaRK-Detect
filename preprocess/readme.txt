---
## Dataset Preprocessing Steps

1.  **Start with Raw Data:** Begin with the **original satellite images** located in the `image` folder and their corresponding **original segmentation masks** in the `mask` folder.

2.  **Generate Road Centerlines (Scribbles):**
    * Use the `full2scribble.py` script to convert the thick, original segmentation masks into thin, single-pixel-wide **road centerline labels** (scribbles).
    * Store these generated scribbles in the `scribble` folder.

3.  **Extract Road Branch Key Points:**
    * Process the road centerline labels (scribbles) from the `scribble` folder using the `find_key_points.py` script.
    * This script identifies and extracts **key points** that represent road branches, intersections, and endpoints.
    * Save these extracted key points as `.mat` files in the `key_points` folder.

4.  **Format Key Points:**
    * Run the `format_transform.py` script on the key points in the `key_points` folder.
    * This step standardizes the format of the key points, ensuring consistency for subsequent processing.
    * Place the newly formatted key points (still as `.mat` files) into the `key_points_final` folder.

5.  **Add Key Point Adjacency Relationships:**
    * Apply the `add_link.py` script to the formatted key points from the `key_points_final` folder.
    * This script establishes **adjacency relationships** between the key points, transforming them into a connected graph structure that represents the road network.
    * Store these key points, now including their network connections, as `.mat` files in the `link_key_points_final` folder.

---
### Verification Steps

* **Test Key Point Accuracy:** Use `test_key_points.py` to visually inspect and verify the accuracy of the extracted key points from the `key_points_final` folder.
* **Test Link Accuracy:** Use `test_link.py` to visually confirm that the adjacency relationships added to the key points are correct and accurately represent the road network.