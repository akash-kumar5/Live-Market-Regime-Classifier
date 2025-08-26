# live_monitor.py
import os
import time
import sys
from datetime import datetime

# --- Import your existing pipeline class ---
# Make sure live_pipeline.py is in the same directory
from main import LiveInferencePipeline 

# --- Configuration (should match your other script) ---
MODEL_FOLDER = "models/"

def run_monitoring_loop():
    """Initializes the pipeline and runs it in a continuous 60-second loop."""
    model_file = os.path.join(MODEL_FOLDER, "lstm_regime_model.keras")
    scaler_file = os.path.join(MODEL_FOLDER, "scaler.joblib")
    meta_file = os.path.join(MODEL_FOLDER, "lstm_model_metadata.json")

    if not all(os.path.exists(f) for f in [model_file, scaler_file, meta_file]):
        print(" Error: Model, scaler, or metadata file not found. Please train the model first.")
        return

    # 1. Initialize the pipeline just once
    pipeline = LiveInferencePipeline(model_file, scaler_file, meta_file)
    
    # Run once immediately without waiting
    pipeline.run_prediction_cycle(fetch_open_candle=True)

    # 2. Start the continuous 1-minute loop
    while True:
        try:
            start_time = time.time()
            
            # --- The Loader Animation ---
            wait_seconds = 60
            # spinner_chars = ['|', '/', '-', '\\']
            print("Waiting for the next minute...")
            # We check for the start of a new minute to stay aligned
            while datetime.now().second != 0:
                # Calculate remaining seconds for a slightly more accurate loader
                remaining = 59 - datetime.now().second
                # char = spinner_chars[datetime.now().second % len(spinner_chars)]
                # Use carriage return '\r' to animate on a single line
                if remaining%5==0:
                    print(f" Next run in {remaining}s... ", end='\r')
                time.sleep(0.2)
            
            # Clear the loader line
            print(" " * 40, end='\r')

            # --- Run the actual prediction cycle ---
            pipeline.run_prediction_cycle(fetch_open_candle=True)

            # --- Ensure the loop takes roughly 60 seconds ---
            # This part is less critical with the alignment check above, but is good practice
            elapsed_time = time.time() - start_time
            sleep_time = max(0, wait_seconds - elapsed_time)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n Monitoring stopped by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred in the monitoring loop: {e}")
            time.sleep(30) # Wait 30s before retrying after an error

if __name__ == "__main__":
    run_monitoring_loop()