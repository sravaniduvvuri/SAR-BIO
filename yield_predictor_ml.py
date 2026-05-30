import os
import re
import glob
import traceback

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance


# ============================================================
# LOGGER
# ============================================================

def write_log(log_file, text):

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

    print(text)


# ============================================================
# FIND CROP COLUMN FLEXIBLY
# ============================================================

def find_crop_column(columns):

    for col in columns:

        if str(col).strip().lower() == "crop":
            return col

    return None


# ============================================================
# MAIN ML PIPELINE
# ============================================================

def run_ml_pipeline(

    gcp_csv,
    zonal_csv_folder,
    output_folder,
    algorithm="Random Forest Regressor",
    progress_callback=None,
    log_file=None
):

    try:

        # ====================================================
        # OUTPUT FOLDER
        # ====================================================

        prediction_output = os.path.join(
            output_folder,
            "Prediction_output"
        )

        os.makedirs(prediction_output, exist_ok=True)

        # ====================================================
        # DETERMINE MODEL TAG
        # ====================================================

        if algorithm == "Random Forest Regressor":

            model_tag = "RF"

        elif algorithm == "Support Vector Machine Regressor":

            model_tag = "SVM"

        else:

            raise ValueError(
                f"Unsupported algorithm selected: {algorithm}"
            )

        write_log(
            log_file,
            f"Selected Algorithm: {algorithm}"
        )

        # ====================================================
        # READ GCP CSV
        # ====================================================

        write_log(log_file, "\nReading GCP CSV...")

        master_df = pd.read_csv(gcp_csv)

        write_log(log_file, str(master_df.head()))

        # ====================================================
        # CREATE Parcel_ID
        # ====================================================

        master_df["Parcel_ID"] = range(
            1,
            len(master_df) + 1
        )

        # ====================================================
        # FIND CROP COLUMN
        # ====================================================

        crop_col = find_crop_column(
            master_df.columns
        )

        if crop_col is None:

            raise ValueError(
                "Crop column not found in GCP CSV."
            )

        write_log(
            log_file,
            f"Crop column detected: {crop_col}"
        )

        # ====================================================
        # FIND DATE COLUMNS
        # ====================================================

        date_columns = []

        for col in master_df.columns:

            if re.match(r"\d+_\d+", str(col)):

                date_columns.append(col)

        write_log(
            log_file,
            f"Date columns found: {date_columns}"
        )

        # ====================================================
        # FIND ZONAL CSV FILES
        # ====================================================

        csv_files = glob.glob(

            os.path.join(
                zonal_csv_folder,
                "zonal_statistics_*.csv"
            )
        )

        write_log(
            log_file,
            f"Total zonal CSV files found: {len(csv_files)}"
        )

        # ====================================================
        # CREATE SAR MASTER TABLE
        # ====================================================

        sar_master = None

        for csv_path in csv_files:

            filename = os.path.basename(csv_path)

            match = re.search(
                r"zonal_statistics_(\d+_\d+)\.csv",
                filename
            )

            if not match:
                continue

            date_tag = match.group(1)

            write_log(
                log_file,
                f"Processing SAR date: {date_tag}"
            )

            df = pd.read_csv(csv_path)

            if "Parcel_ID" not in df.columns:

                df["Parcel_ID"] = range(
                    1,
                    len(df) + 1
                )

            required_sar_cols = [

                "VV_mean",
                "VH_mean",
                "RVI_mean",
                "CR_mean"
            ]

            missing_cols = [

                col for col in required_sar_cols

                if col not in df.columns
            ]

            if len(missing_cols) > 0:

                write_log(
                    log_file,
                    f"Missing SAR columns: {missing_cols}"
                )

                continue

            df = df[[

                "Parcel_ID",

                "VV_mean",

                "VH_mean",

                "RVI_mean",

                "CR_mean"
            ]]

            df = df.rename(columns={

                "VV_mean": f"VV_{date_tag}",

                "VH_mean": f"VH_{date_tag}",

                "RVI_mean": f"RVI_{date_tag}",

                "CR_mean": f"CR_{date_tag}"
            })

            if sar_master is None:

                sar_master = df

            else:

                sar_master = sar_master.merge(

                    df,

                    on="Parcel_ID",

                    how="outer"
                )

        # ====================================================
        # MERGE TRAINING TABLE
        # ====================================================

        training_df = master_df.merge(

            sar_master,

            on="Parcel_ID",

            how="inner"
        )

        training_df = training_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        write_log(
            log_file,
            f"Training samples: {len(training_df)}"
        )

        # ====================================================
        # RESULTS STORAGE
        # ====================================================

        results_list = []

        total_steps = (

            len(training_df[crop_col].unique())

            * len(date_columns)
        )

        current_step = 0

        # ====================================================
        # CROP LOOP
        # ====================================================

        for crop_name in training_df[crop_col].unique():

            write_log(
                log_file,
                f"\nPROCESSING CROP: {crop_name}"
            )

            crop_df = training_df[

                training_df[crop_col] == crop_name

            ].copy()

            # =================================================
            # DATE LOOP
            # =================================================

            for date_tag in date_columns:

                current_step += 1

                write_log(
                    log_file,
                    f"PROCESSING DATE: {date_tag}"
                )

                biomass_col = date_tag

                feature_cols = [

                    f"VV_{date_tag}",

                    f"VH_{date_tag}",

                    f"RVI_{date_tag}",

                    f"CR_{date_tag}"
                ]

                missing_features = [

                    col for col in feature_cols

                    if col not in crop_df.columns
                ]

                if len(missing_features) > 0:

                    write_log(
                        log_file,
                        f"Missing features: {missing_features}"
                    )

                    continue

                model_df = crop_df[[

                    biomass_col

                ] + feature_cols].copy()

                model_df = model_df.dropna()

                if len(model_df) < 5:

                    write_log(
                        log_file,
                        "Too few samples."
                    )

                    continue

                # =============================================
                # OUTLIER REMOVAL
                # =============================================

                Q1 = model_df[biomass_col].quantile(0.25)

                Q3 = model_df[biomass_col].quantile(0.75)

                IQR = Q3 - Q1

                mask = (

                    (model_df[biomass_col] >= Q1 - 1.5 * IQR)

                    &

                    (model_df[biomass_col] <= Q3 + 1.5 * IQR)
                )

                model_df = model_df[mask]

                if len(model_df) < 5:

                    write_log(
                        log_file,
                        "Too few samples after outlier removal."
                    )

                    continue

                # =============================================
                # FEATURES + TARGET
                # =============================================

                X = model_df[feature_cols]

                y = model_df[biomass_col]

                scaler = StandardScaler()

                X_scaled = scaler.fit_transform(X)

                # =============================================
                # MODEL SELECTION
                # =============================================

                if algorithm == "Random Forest Regressor":

                    model = RandomForestRegressor(

                        n_estimators=300,

                        max_depth=4,

                        min_samples_leaf=2,

                        min_samples_split=4,

                        max_features='sqrt',

                        random_state=42
                    )

                elif algorithm == "Support Vector Machine Regressor":

                    model = SVR(

                        kernel='rbf',

                        C=10,

                        gamma='scale',

                        epsilon=0.1
                    )

                else:

                    raise ValueError(
                        f"Unsupported algorithm selected: {algorithm}"
                    )

                # =============================================
                # LOOCV
                # =============================================

                loo = LeaveOneOut()

                observed = []

                predicted = []

                for train_index, test_index in loo.split(X_scaled):

                    X_train = X_scaled[train_index]

                    X_test = X_scaled[test_index]

                    y_train = y.iloc[train_index]

                    y_test = y.iloc[test_index]

                    model.fit(
                        X_train,
                        y_train
                    )

                    y_pred = model.predict(X_test)

                    observed.append(
                        y_test.values[0]
                    )

                    predicted.append(
                        y_pred[0]
                    )

                # =============================================
                # METRICS
                # =============================================

                r2 = r2_score(
                    observed,
                    predicted
                )

                rmse = np.sqrt(

                    mean_squared_error(
                        observed,
                        predicted
                    )
                )

                write_log(
                    log_file,
                    f"R2={r2:.3f} RMSE={rmse:.3f}"
                )

                # =============================================
                # FINAL TRAIN
                # =============================================

                model.fit(
                    X_scaled,
                    y
                )

                # =============================================
                # FEATURE IMPORTANCE
                # =============================================

                if algorithm == "Random Forest Regressor":

                    importance_df = pd.DataFrame({

                        "Feature": feature_cols,

                        "Importance": model.feature_importances_
                    })

                elif algorithm == "Support Vector Machine Regressor":

                    write_log(
                        log_file,
                        "Calculating permutation importance..."
                    )

                    perm_importance = permutation_importance(

                        model,

                        X_scaled,

                        y,

                        n_repeats=20,

                        random_state=42
                    )

                    importance_df = pd.DataFrame({

                        "Feature": feature_cols,

                        "Importance":
                        perm_importance.importances_mean
                    })

                importance_df = importance_df.sort_values(

                    by="Importance",

                    ascending=False
                )

                # =============================================
                # OBSERVED VS PREDICTED
                # =============================================

                pred_df = pd.DataFrame({

                    "Observed": observed,

                    "Predicted": predicted
                })

                # =============================================
                # OUTPUT FILES
                # =============================================

                importance_csv = os.path.join(

                    prediction_output,

                    f"{crop_name}_{date_tag}_{model_tag}_Importance.csv"
                )

                pred_csv = os.path.join(

                    prediction_output,

                    f"{crop_name}_{date_tag}_{model_tag}_Observed_vs_Predicted.csv"
                )

                importance_df.to_csv(

                    importance_csv,

                    index=False
                )

                pred_df.to_csv(

                    pred_csv,

                    index=False
                )

                results_list.append({

                    "Crop": crop_name,

                    "Date": date_tag,

                    "Samples": len(model_df),

                    "R2": r2,

                    "RMSE": rmse
                })

                # =============================================
                # ML PROGRESS
                # =============================================

                if progress_callback:

                    progress = int(

                        (current_step / total_steps)

                        * 100
                    )

                    progress_callback(progress)

        # ====================================================
        # FINAL RESULTS
        # ====================================================

        results_df = pd.DataFrame(results_list)

        final_csv = os.path.join(

            prediction_output,

            f"Final_Temporal_Biomass_{model_tag}_Results.csv"
        )

        results_df.to_csv(

            final_csv,

            index=False
        )

        write_log(
            log_file,
            "ML pipeline completed successfully."
        )

        return True

    except Exception as e:

        traceback.print_exc()

        write_log(
            log_file,
            str(e)
        )

        raise