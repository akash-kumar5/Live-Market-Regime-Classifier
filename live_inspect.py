import os
import time
from datetime import datetime
from main import LiveInferencePipeline 

MODEL_FOLDER = "models/"

def run_monitoring_loop():
    """Initializes the pipeline and runs it in a continuous loop for real-time inspection."""
    model_file = os.path.join(MODEL_FOLDER, "lstm_regime_model.keras")
    scaler_file = os.path.join(MODEL_FOLDER, "scaler.joblib")
    meta_file = os.path.join(MODEL_FOLDER, "lstm_model_metadata.json")

    if not all(os.path.exists(f) for f in [model_file, scaler_file, meta_file]):
        print("Error: Model, scaler, or metadata file not found. Please train the model first.")
        return

    pipeline = LiveInferencePipeline(model_file, scaler_file, meta_file)
    
    while True:
        try:
            # Run with open candle to get the most up-to-date data
            pipeline.run_prediction_cycle(fetch_open_candle=True)

            print("Waiting for 1 minute before next inspection...")
            time.sleep(60)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred in the monitoring loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    run_monitoring_loop()
