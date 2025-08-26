import os
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Qiskit Imports ---
print("Checkpoint 1: Importing Qiskit libraries...")
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
print("Checkpoint 2: Qiskit libraries imported successfully.")

# =============================================================================
#  CONFIGURATION
# =============================================================================
# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# =============================================================================
#  HELPER FUNCTION: Convert Matplotlib figure to Base64 PNG
# =============================================================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# =============================================================================
#  LOCAL SIMULATION LOGIC (deterministic, no Aer dependency)
# =============================================================================
def run_local_simulation(message: str, shots: int = 1024):
    print("--- Running Local Simulation (deterministic) ---")
    q = QuantumRegister(2, 'q')
    c = ClassicalRegister(2, 'c')
    circ = QuantumCircuit(q, c)
    
    circ.h(q[0]); circ.cx(q[0], q[1]); circ.barrier()
    if message == '01': circ.x(q[0])
    elif message == '10': circ.z(q[0])
    elif message == '11': circ.z(q[0]); circ.x(q[0])
    circ.barrier()
    circ.cx(q[0], q[1]); circ.h(q[0]); circ.barrier()
    circ.measure(q[0], c[0]); circ.measure(q[1], c[1])

    # Deterministic counts without simulation (ideal noiseless superdense coding)
    counts = {'00': 0, '01': 0, '10': 0, '11': 0}
    counts[message] = shots
    success_rate = 1.0

    circuit_fig = circ.draw(output='mpl', style='iqp')
    hist_fig = plot_histogram(counts, title=f"Local (Ideal) (Message: {message})")

    return {
        "counts": counts,
        "success_rate": success_rate,
        "circuit_image_b64": fig_to_base64(circuit_fig),
        "histogram_image_b64": fig_to_base64(hist_fig),
    }

# =============================================================================
#  IBM QUANTUM PLATFORM LOGIC (lazy import; may not be available)
# =============================================================================
def run_ibm_simulation(message: str, shots: int = 1024):
    print("--- Running IBM Simulation ---")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
        IBM_QUANTUM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN", "")
        IBM_INSTANCE = os.getenv("IBM_INSTANCE", "")

        service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                       token=IBM_QUANTUM_TOKEN,
                                       instance=IBM_INSTANCE)
        backend = service.least_busy(simulator=False, operational=True)
        print(f"Using backend: {backend.name}")
    except Exception as e:
        print(f"ERROR during IBM connection: {e}")
        return {"error": f"IBM Quantum Connection Error: {str(e)}"}

    q = QuantumRegister(2, "q")
    c = ClassicalRegister(2, "c")
    qc = QuantumCircuit(q, c)

    qc.h(q[0]); qc.cx(q[0], q[1])
    if message == "01": qc.x(q[0])
    elif message == "10": qc.z(q[0])
    elif message == "11": qc.z(q[0]); qc.x(q[0])
    qc.cx(q[0], q[1]); qc.h(q[0]); qc.measure(q, c)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circ = pm.run(qc)

    sampler = Sampler(backend)
    job = sampler.run([isa_circ], shots=shots)
    job_id = job.job_id()
    res = job.result()
    pub = res[0]

    # Raw counts from backend
    counts_raw = getattr(pub.data, c.name).get_counts()

    # Fix endian (reverse keys like local)
    counts = {'00': 0, '01': 0, '10': 0, '11': 0}
    for k, v in counts_raw.items():
        counts[k[::-1]] += v

    circuit_fig = qc.draw(output='mpl', style='iqp', idle_wires=False)
    hist_fig = plot_histogram(counts, title=f"IBM Backend: {backend.name} (Message: {message})")

    return {
        "job_id": job_id,
        "backend_name": backend.name,
        "counts": counts,
        "circuit_image_b64": fig_to_base64(circuit_fig),
        "histogram_image_b64": fig_to_base64(hist_fig),
    }

# =============================================================================
#  API ENDPOINT
# =============================================================================
@app.route('/api/run_simulation', methods=['POST'])
def run_simulation_endpoint():
    try:
        data = request.get_json()
        message = data.get('message')
        target = data.get('target')

        if not message or message not in ['00', '01', '10', '11']:
            return jsonify({"error": "Invalid message provided."}), 400
        if not target or target not in ['local', 'ibm']:
            return jsonify({"error": "Invalid target provided."}), 400
        
        if target == 'local':
            result = run_local_simulation(message)
        else:
            result = run_ibm_simulation(message)
        
        # If IBM path returned an error, surface it as error HTTP to keep UI consistent
        if isinstance(result, dict) and result.get('error'):
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500

# =============================================================================
#  RUN THE APP
# =============================================================================
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5000)
