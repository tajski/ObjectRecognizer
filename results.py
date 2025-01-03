import pandas as pd
import os


class ResultsAnalyzer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.results_file = os.path.join(folder_path, 'results.csv')

    def load_results(self):
        return pd.read_csv(self.results_file)

    def summarize_metrics(self):
        df = self.load_results()
        metrics = {
            "mAP@0.5": df["metrics/mAP50(B)"].iloc[-1],
            "mAP@0.5:0.95": df["metrics/mAP50-95(B)"].iloc[-1],
            "Precyzja": df["metrics/precision(B)"].iloc[-1],
            "Czułość": df["metrics/recall(B)"].iloc[-1],
            "Strata pola": df["val/box_loss"].iloc[-1],
            "Błąd klasyfikacji": df["val/cls_loss"].iloc[-1],
            "Strata cech głębokich": df["val/dfl_loss"].iloc[-1]
        }
        return metrics

    def display_metrics(self):
        metrics = self.summarize_metrics()
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    folders = {
        "coco128": "YOLOv8-Experiments/yolo_default",
        "custom_data": "YOLOv8-Experiments/custom"
    }

    for name, path in folders.items():
        print(f"\n=== Metryki wydajności modelu dla danych '{name}' ===")
        analyzer = ResultsAnalyzer(folder_path=path)
        analyzer.display_metrics()