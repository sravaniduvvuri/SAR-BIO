import os
import re

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.features import rasterize

from rasterstats import zonal_stats


# ============================================================
# LOGGER
# ============================================================

def write_log(log_file, text):

    if log_file:

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    print(text)


# ============================================================
# SAFE CRS HANDLING
# ============================================================

def safe_reproject(gdf, target_crs):

    # Shapefile CRS required
    if gdf.crs is None:
        raise ValueError("Shapefile CRS is missing.")

    # Raster CRS missing -> assume already aligned
    if target_crs is None:

        write_log(
            None,
            "WARNING: Raster CRS missing. Using shapefile CRS directly."
        )

        return gdf

    # Same CRS -> skip reprojection
    if gdf.crs == target_crs:
        return gdf

    # Reproject only if needed
    return gdf.to_crs(target_crs)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_full_zonal_pipeline(
    vv_files,
    vh_files,
    shapefile_path,
    output_folder,
    progress_callback=None,
    log_file=None
):

    # ========================================================
    # OUTPUT ROOT
    # ========================================================

    os.makedirs(output_folder, exist_ok=True)

    write_log(
        log_file,
        f"Output folder created: {output_folder}"
    )

    # ========================================================
    # READ SHAPEFILE
    # ========================================================

    write_log(
        log_file,
        f"Reading shapefile: {shapefile_path}"
    )

    gdf = gpd.read_file(shapefile_path)

    total = len(vv_files)

    csv_all = []

    write_log(
        log_file,
        f"Total VV rasters: {total}"
    )

    # ========================================================
    # BUILD VH MAP
    # ========================================================

    vh_map = {}

    for vh in vh_files:

        m = re.search(
            r"VH_(.*?)\.tif",
            os.path.basename(vh),
            re.IGNORECASE
        )

        if m:
            vh_map[m.group(1)] = vh

    write_log(
        log_file,
        f"Total VH rasters mapped: {len(vh_map)}"
    )

    # ========================================================
    # OUTPUT FOLDERS
    # ========================================================

    out_rvi = os.path.join(output_folder, "RVI")

    out_cr = os.path.join(output_folder, "CR")

    out_csv = os.path.join(output_folder, "CSV")

    out_vv = os.path.join(output_folder, "VV")

    out_vh = os.path.join(output_folder, "VH")

    for p in [out_rvi, out_cr, out_csv, out_vv, out_vh]:

        os.makedirs(p, exist_ok=True)

        write_log(
            log_file,
            f"Created folder: {p}"
        )

    # ========================================================
    # PROCESS LOOP
    # ========================================================

    for i, vv_path in enumerate(vv_files):

        filename = os.path.basename(vv_path)

        match = re.search(
            r"VV_(.*?)\.tif",
            filename,
            re.IGNORECASE
        )

        if not match:

            write_log(
                log_file,
                f"Skipping invalid VV file: {filename}"
            )

            continue

        date_tag = match.group(1)

        vh_path = vh_map.get(date_tag)

        if not vh_path or not os.path.exists(vh_path):

            write_log(
                log_file,
                f"Skipping missing VH for {date_tag}"
            )

            continue

        write_log(
            log_file,
            f"\nProcessing date: {date_tag}"
        )

        # ====================================================
        # READ VV
        # ====================================================

        write_log(
            log_file,
            f"Reading VV raster: {vv_path}"
        )

        with rasterio.open(vv_path) as vv_src:

            vv = vv_src.read(1, masked=True).astype("float32")

            meta = vv_src.meta.copy()

            # IMPORTANT FIX
            meta.update(
                driver="GTiff",
                compress="lzw",
                dtype="float32"
            )

            crs = vv_src.crs

            transform = vv_src.transform

            out_shape = (
                vv_src.height,
                vv_src.width
            )

        # ====================================================
        # CRS MATCH
        # ====================================================

        write_log(
            log_file,
            "Matching CRS..."
        )

        gdf_proj = safe_reproject(gdf, crs)

        # ====================================================
        # READ VH
        # ====================================================

        write_log(
            log_file,
            f"Reading VH raster: {vh_path}"
        )

        with rasterio.open(vh_path) as vh_src:

            vh = vh_src.read(1, masked=True).astype("float32")

        # ====================================================
        # CLEAN INPUTS
        # ====================================================

        vv = np.clip(vv, -40, 10)

        vh = np.clip(vh, -40, 10)

        # ====================================================
        # LINEAR SCALE
        # ====================================================

        vv_linear = 10 ** (vv / 10)

        vh_linear = 10 ** (vh / 10)

        epsilon = 1e-10

        # ====================================================
        # CALCULATE INDICES
        # ====================================================

        write_log(
            log_file,
            "Calculating RVI and CR..."
        )

        rvi = (
            4 * vh_linear
        ) / (
            vv_linear + vh_linear + epsilon
        )

        cr = vh_linear / (
            vv_linear + epsilon
        )

        # ====================================================
        # CLEAN VALUES
        # ====================================================

        rvi = np.clip(rvi, 0, 1)

        valid_mask = vv_linear > 0.001

        rvi[~valid_mask] = np.nan

        cr[~valid_mask] = np.nan

        # ====================================================
        # SAVE RVI + CR
        # ====================================================

        rvi_path = os.path.join(
            out_rvi,
            f"RVI_{date_tag}.tif"
        )

        cr_path = os.path.join(
            out_cr,
            f"CR_{date_tag}.tif"
        )

        with rasterio.open(rvi_path, "w", **meta) as dst:

            dst.write(
                rvi.astype("float32"),
                1
            )

        write_log(
            log_file,
            f"Saved RVI raster: {rvi_path}"
        )

        with rasterio.open(cr_path, "w", **meta) as dst:

            dst.write(
                cr.astype("float32"),
                1
            )

        write_log(
            log_file,
            f"Saved CR raster: {cr_path}"
        )

        # ====================================================
        # ZONAL STATISTICS
        # ====================================================

        write_log(
            log_file,
            "Running zonal statistics..."
        )

        rasters = {
            "VV": vv_path,
            "VH": vh_path,
            "RVI": rvi_path,
            "CR": cr_path
        }

        results = gdf_proj.copy()

        for key, path in rasters.items():

            write_log(
                log_file,
                f"Calculating zonal mean for {key}"
            )

            stats = zonal_stats(
                gdf_proj,
                path,
                stats="mean",
                nodata=np.nan
            )

            results[key + "_mean"] = [
                s["mean"] for s in stats
            ]

        # ====================================================
        # SAVE CSV
        # ====================================================

        csv_path = os.path.join(
            out_csv,
            f"zonal_statistics_{date_tag}.csv"
        )

        results.drop(
            columns="geometry"
        ).to_csv(
            csv_path,
            index=False
        )

        csv_all.append(
            results.drop(columns="geometry")
        )

        write_log(
            log_file,
            f"CSV saved: {csv_path}"
        )

        # ====================================================
        # RASTERIZE MEAN VALUES
        # ====================================================

        raster_outputs = {
            "VV_mean": out_vv,
            "VH_mean": out_vh,
            "RVI_mean": out_rvi,
            "CR_mean": out_cr
        }

        for var, folder in raster_outputs.items():

            write_log(
                log_file,
                f"Rasterizing {var}"
            )

            shapes = (
                (geom, value)
                for geom, value in zip(
                    results.geometry,
                    results[var]
                )
            )

            rasterized = rasterize(
                shapes=shapes,
                out_shape=out_shape,
                transform=transform,
                fill=np.nan,
                dtype="float32"
            )

            out_path = os.path.join(
                folder,
                f"{var}_{date_tag}.tif"
            )

            with rasterio.open(
                out_path,
                "w",
                **meta
            ) as dst:

                dst.write(
                    rasterized.astype("float32"),
                    1
                )

            write_log(
                log_file,
                f"Saved rasterized output: {out_path}"
            )

        # ====================================================
        # PROGRESS UPDATE
        # ====================================================

        if progress_callback:

            progress_callback(
                int((i + 1) / total * 100)
            )

            write_log(
                log_file,
                f"Progress updated: {int((i + 1) / total * 100)}%"
            )

    # ========================================================
    # FINAL MERGED CSV
    # ========================================================

    if csv_all:

        final_df = pd.concat(
            csv_all,
            ignore_index=True
        )

        final_csv = os.path.join(
            out_csv,
            "FINAL_ALL_DATES.csv"
        )

        final_df.to_csv(
            final_csv,
            index=False
        )

        write_log(
            log_file,
            f"Final merged CSV saved: {final_csv}"
        )

    write_log(
        log_file,
        "ALL FILES PROCESSED SUCCESSFULLY"
    )

    return True