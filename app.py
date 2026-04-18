import os
import subprocess
from flask import Flask, send_from_directory, render_template_string

app = Flask(__name__)

# Directory where run_all.py generates index.html
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

@app.route("/")
def index():
    # If index.html doesn't exist yet, run the simulation
    if not os.path.exists(os.path.join(OUTPUT_DIR, "index.html")):
        run_simulation()
    
    return send_from_directory(OUTPUT_DIR, "index.html")

@app.route("/run")
def run():
    # Force a re-run of the simulation
    try:
        result = run_simulation()
        if result == 0:
            return "Simulation completed successfully! <a href='/'>View Dashboard</a>"
        else:
            return "Simulation failed. Check server logs."
    except Exception as e:
        return f"Error running simulation: {str(e)}"

def run_simulation():
    print("Starting simulation...")
    # Run run_all.py as a subprocess
    process = subprocess.Popen(["python", "run_all.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Simulation Error:\n{stderr.decode()}")
    else:
        print("Simulation Finished.")
    
    return process.returncode

if __name__ == "__main__":
    # Ensure outputs directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
