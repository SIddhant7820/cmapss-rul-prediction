import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from src.utils import get_logger, load_config


class CMAPSSTransformer:

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)

        prep = self.config["preprocessing"]
        self.max_rul = prep.get("max_rul", 125)
        self.window_size_default = prep.get("window_size", 30)
        self.drop_sensors = prep.get("drop_sensors", [])
        self.op_condition_clusters = prep.get("op_condition_clusters", {})
        self.use_cluster_normalization = prep.get("use_cluster_normalization", False)
        self.seed = self.config.get("model", {}).get("seed", 42)

        self.processed_dir = Path(self.config["dataset"]["processed_path"])
        self.model_save_dir = Path(self.config["paths"]["model_save_path"])

        # will be set during normalize()
        self.feature_cols = []
        self._kmeans_models = {}

    def compute_rul(self, train_df: pd.DataFrame) -> pd.DataFrame:
        # piecewise linear RUL capping
        df = train_df.copy()
        max_cycle = df.groupby("unit_id")["cycle"].transform("max")
        df["RUL"] = (max_cycle - df["cycle"]).clip(upper=self.max_rul).astype(np.float32)
        self.logger.info("RUL computed and capped at %d", self.max_rul)
        return df

    def drop_low_variance_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        # drop sensors that dont contribute much
        to_drop = [c for c in self.drop_sensors if c in df.columns]
        self.logger.info("Dropping sensors: %s", to_drop)
        return df.drop(columns=to_drop)

    def get_operating_regime(self, df: pd.DataFrame, subset: str) -> pd.DataFrame:
        out = df.copy()
        clusters = self.op_condition_clusters.get(subset, 1)

        if clusters == 1:
            out["regime"] = 0
            return out

        # use kmeans to find operating regimes for FD002/FD004
        features = out[["op1", "op2", "op3"]].astype(float).values
        if subset not in self._kmeans_models:
            self.logger.info("Fitting KMeans with %d clusters for %s", clusters, subset)
            kmeans = KMeans(n_clusters=clusters, random_state=self.seed, n_init="auto")
            out["regime"] = kmeans.fit_predict(features)
            self._kmeans_models[subset] = kmeans
        else:
            out["regime"] = self._kmeans_models[subset].predict(features)

        return out

    def normalize(self, train_df: pd.DataFrame, test_df: pd.DataFrame, subset: str):
        clusters = self.op_condition_clusters.get(subset, 1)
        sensor_cols = sorted([c for c in train_df.columns if c.startswith("s")])
        self.feature_cols = sensor_cols

        train_out = train_df.copy()
        test_out = test_df.copy()

        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = self.model_save_dir / f"scaler_{subset}.pkl"

        use_cluster = self.use_cluster_normalization and clusters > 1

        if use_cluster:
            # fit one scaler per operating regime
            scalers = {}
            for r in range(clusters):
                mask = train_out["regime"] == r
                scaler = MinMaxScaler()
                if mask.any():
                    scaler.fit(train_out.loc[mask, sensor_cols].astype(float))
                else:
                    # fallback if regime missing in train
                    scaler.fit(train_out[sensor_cols].astype(float))
                scalers[r] = scaler

            for r, scaler in scalers.items():
                tr_mask = train_out["regime"] == r
                te_mask = test_out["regime"] == r
                if tr_mask.any():
                    train_out.loc[tr_mask, sensor_cols] = scaler.transform(
                        train_out.loc[tr_mask, sensor_cols].astype(float)
                    )
                if te_mask.any():
                    test_out.loc[te_mask, sensor_cols] = scaler.transform(
                        test_out.loc[te_mask, sensor_cols].astype(float)
                    )

            joblib.dump({"mode": "cluster", "scalers": scalers, "sensor_cols": sensor_cols}, scaler_path)

        else:
            # single scaler for FD001/FD003
            # TODO: try StandardScaler here and compare results
            scaler = MinMaxScaler()
            scaler.fit(train_out[sensor_cols].astype(float))
            train_out[sensor_cols] = scaler.transform(train_out[sensor_cols].astype(float))
            test_out[sensor_cols] = scaler.transform(test_out[sensor_cols].astype(float))

            joblib.dump({"mode": "global", "scaler": scaler, "sensor_cols": sensor_cols}, scaler_path)

        self.logger.info("Scaler saved to %s", scaler_path)
        return train_out, test_out

    def create_sequences(self, df: pd.DataFrame, window_size: int = None):
        w = window_size or self.window_size_default
        X_list, y_list = [], []

        for _, grp in df.groupby("unit_id"):
            grp = grp.sort_values("cycle")
            values = grp[self.feature_cols].astype(np.float32).values
            rul = grp["RUL"].astype(np.float32).values

            # skip engines with too few cycles
            if len(grp) < w:
                continue

            for start in range(len(grp) - w + 1):
                X_list.append(values[start:start + w])
                y_list.append(rul[start + w - 1])

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.float32)
        return X, y

    def prepare_test_sequences(self, test_df: pd.DataFrame, rul_df: pd.DataFrame, window_size: int = None):
        w = window_size or self.window_size_default
        units = sorted(test_df["unit_id"].astype(int).unique())
        rul_values = rul_df["RUL"].astype(np.float32).values

        X_list, y_list = [], []

        for i, uid in enumerate(units):
            grp = test_df[test_df["unit_id"] == uid].sort_values("cycle")

            # just take the last window for each engine
            last = grp.tail(w)
            x_seq = last[self.feature_cols].astype(np.float32).values

            # pad if engine has fewer cycles than window size
            if len(last) < w:
                pad = np.zeros((w - len(last), len(self.feature_cols)), dtype=np.float32)
                x_seq = np.vstack([pad, x_seq])

            X_list.append(x_seq)
            y_list.append(rul_values[i])

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.float32)
        return X, y

    def save_processed(self, X_train, y_train, X_test, y_test, subset: str):
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.processed_dir / f"X_train_{subset}.npy", X_train)
        np.save(self.processed_dir / f"y_train_{subset}.npy", y_train)
        np.save(self.processed_dir / f"X_test_{subset}.npy", X_test)
        np.save(self.processed_dir / f"y_test_{subset}.npy", y_test)

        self.logger.info(
            "Saved processed data for %s → X_train%s y_train%s X_test%s y_test%s",
            subset, X_train.shape, y_train.shape, X_test.shape, y_test.shape
        )

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame, rul_df: pd.DataFrame, subset: str):
        self.logger.info("Starting transformation pipeline for %s", subset)

        train_df = self.compute_rul(train_df)

        train_df = self.drop_low_variance_sensors(train_df)
        test_df = self.drop_low_variance_sensors(test_df.copy())

        train_df = self.get_operating_regime(train_df, subset)
        test_df = self.get_operating_regime(test_df, subset)

        train_df, test_df = self.normalize(train_df, test_df, subset)

        X_train, y_train = self.create_sequences(train_df)
        X_test, y_test = self.prepare_test_sequences(test_df, rul_df)

        self.save_processed(X_train, y_train, X_test, y_test, subset)

        self.logger.info("Transformation done for %s", subset)
        return X_train, y_train, X_test, y_test