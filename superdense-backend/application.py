import os
import base64
import io
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
import random

# --- Qiskit Imports ---
print("Checkpoint 1: Importing Qiskit libraries...")
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, circuit_drawer
from qiskit.quantum_info import random_statevector, Statevector
print("Checkpoint 2: Qiskit libraries imported successfully.")

# =============================================================================
#  CONFIGURATION
# =============================================================================
matplotlib.use('Agg')

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================
def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_random_bits(n):
    """Generate n random bits"""
    return [random.randint(0, 1) for _ in range(n)]

def calculate_qber(alice_bits, bob_bits):
    """Calculate Quantum Bit Error Rate"""
    if len(alice_bits) != len(bob_bits) or len(alice_bits) == 0:
        return 0.0
    
    errors = sum(1 for a, b in zip(alice_bits, bob_bits) if a != b)
    return errors / len(alice_bits)

# =============================================================================
#  E91 QKD PROTOCOL SIMULATION
# =============================================================================
def simulate_e91_qkd(num_pairs=50, has_eve=False):
    """
    Simulate E91 QKD Protocol
    Returns: dict with key, QBER, measurements, circuit image, histogram data
    """
    print(f"--- Running E91 QKD Simulation with {num_pairs} pairs, Eve: {has_eve} ---")
    
    # Alice and Bob's measurement bases (random)
    alice_bases = generate_random_bits(num_pairs)  # 0: Z basis, 1: X basis
    bob_bases = generate_random_bits(num_pairs)
    
    # For E91, we create entangled pairs and measure
    alice_results = []
    bob_results = []
    
    # Create a sample E91 circuit for visualization
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(qr, cr, name="E91_QKD")
    
    # Create entangled pair (Bell state)
    circuit.h(qr[0])
    circuit.cx(qr[0], qr[1])
    
    # Add measurement in different bases (example)
    circuit.barrier()
    if alice_bases[0] == 1:  # X basis measurement
        circuit.h(qr[0])
    if bob_bases[0] == 1:   # X basis measurement
        circuit.h(qr[1])
    
    circuit.measure(qr[0], cr[0])
    circuit.measure(qr[1], cr[1])
    
    # Simulate measurements
    backend = AerSimulator()
    
    for i in range(num_pairs):
        # Create entangled pair circuit for this measurement
        temp_qr = QuantumRegister(2, 'q')
        temp_cr = ClassicalRegister(2, 'c')
        temp_circuit = QuantumCircuit(temp_qr, temp_cr)
        
        # Bell state
        temp_circuit.h(temp_qr[0])
        temp_circuit.cx(temp_qr[0], temp_qr[1])
        
        # Apply measurement bases
        if alice_bases[i] == 1:  # X basis
            temp_circuit.h(temp_qr[0])
        if bob_bases[i] == 1:   # X basis
            temp_circuit.h(temp_qr[1])
            
        temp_circuit.measure(temp_qr[0], temp_cr[0])
        temp_circuit.measure(temp_qr[1], temp_cr[1])
        
        # Run simulation
        job = backend.run(temp_circuit, shots=1)
        result = job.result()
        counts = result.get_counts(temp_circuit)
        
        # Get the measurement result
        measurement = list(counts.keys())[0]
        alice_results.append(int(measurement[1]))  # Alice's bit
        bob_results.append(int(measurement[0]))    # Bob's bit
    
    # Sift keys (keep only matching bases)
    sifted_alice = []
    sifted_bob = []
    matching_indices = []
    
    for i in range(num_pairs):
        if alice_bases[i] == bob_bases[i]:
            sifted_alice.append(alice_results[i])
            sifted_bob.append(bob_results[i])
            matching_indices.append(i)
    
    # Simulate Eve's interference if requested
    if has_eve:
        # Eve introduces errors by intercepting and re-sending
        eve_error_rate = 0.25  # 25% error due to Eve
        for i in range(len(sifted_bob)):
            if random.random() < eve_error_rate:
                sifted_bob[i] = 1 - sifted_bob[i]  # Flip bit
    
    # Calculate QBER
    qber = calculate_qber(sifted_alice, sifted_bob)
    
    # Generate final key (first half for demonstration)
    key_length = len(sifted_alice) // 2
    final_key = sifted_alice[:key_length] if key_length > 0 else []
    
    # Create histogram data
    histogram_data = [
        {"name": "Matching Bases", "alice": len(matching_indices), "bob": len(matching_indices)},
        {"name": "Key Bits Generated", "alice": len(final_key), "bob": len(final_key)},
        {"name": "Discarded", "alice": num_pairs - len(matching_indices), "bob": num_pairs - len(matching_indices)}
    ]
    
    # Generate circuit image
    circuit_fig = circuit.draw(output='mpl', style='iqp')
    circuit_image = fig_to_base64(circuit_fig)
    
    # Create measurement table
    measurements_table = []
    for i in range(min(10, len(matching_indices))):  # Show first 10 matching measurements
        idx = matching_indices[i]
        measurements_table.append({
            "index": idx,
            "alice_basis": "X" if alice_bases[idx] == 1 else "Z",
            "bob_basis": "X" if bob_bases[idx] == 1 else "Z",
            "alice_result": sifted_alice[i] if i < len(sifted_alice) else 0,
            "bob_result": sifted_bob[i] if i < len(sifted_bob) else 0
        })
    
    return {
        "key": ''.join(map(str, final_key)),
        "qber": qber,
        "qber_percentage": round(qber * 100, 2),
        "measurements_table": measurements_table,
        "circuit_image": circuit_image,
        "histogram_data": histogram_data,
        "num_matching": len(matching_indices),
        "key_length": len(final_key)
    }

# =============================================================================
#  SUPERDENSE CODING WITH QKD KEYS
# =============================================================================
def simulate_superdense_with_qkd(message, qkd_key, has_eve=False):
    """
    Simulate superdense coding using QKD-generated keys
    """
    print(f"--- Running Superdense Coding with message: {message}, Eve: {has_eve} ---")
    
    if len(qkd_key) < 2:
        return {"error": "QKD key too short for encryption"}
    
    # Convert message to binary
    message_bits = [int(b) for b in message]
    
    # Encrypt using XOR with QKD key
    encrypted_bits = [(message_bits[i] ^ int(qkd_key[i % len(qkd_key)])) for i in range(len(message_bits))]
    encrypted_message = ''.join(map(str, encrypted_bits))
    
    # Create superdense coding circuit without Eve
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    circuit_clean = QuantumCircuit(qr, cr, name=f"Superdense_{message}_Clean")
    
    # Bell state preparation
    circuit_clean.h(qr[0])
    circuit_clean.cx(qr[0], qr[1])
    circuit_clean.barrier()
    
    # Encode message
    if message == '01':
        circuit_clean.x(qr[0])
    elif message == '10':
        circuit_clean.z(qr[0])
    elif message == '11':
        circuit_clean.z(qr[0])
        circuit_clean.x(qr[0])
    
    circuit_clean.barrier()
    # Bell measurement
    circuit_clean.cx(qr[0], qr[1])
    circuit_clean.h(qr[0])
    circuit_clean.measure(qr, cr)
    
    # Create circuit with Eve's interference
    circuit_eve = QuantumCircuit(qr, cr, name=f"Superdense_{message}_Eve")
    
    # Bell state preparation
    circuit_eve.h(qr[0])
    circuit_eve.cx(qr[0], qr[1])
    circuit_eve.barrier()
    
    # Eve's interference (adds noise)
    if has_eve:
        circuit_eve.rx(0.1, qr[0])  # Small rotation
        circuit_eve.rz(0.1, qr[1])  # Small rotation
    
    # Encode message
    if message == '01':
        circuit_eve.x(qr[0])
    elif message == '10':
        circuit_eve.z(qr[0])
    elif message == '11':
        circuit_eve.z(qr[0])
        circuit_eve.x(qr[0])
    
    circuit_eve.barrier()
    # Bell measurement
    circuit_eve.cx(qr[0], qr[1])
    circuit_eve.h(qr[0])
    circuit_eve.measure(qr, cr)
    
    # Simulate both circuits
    backend = AerSimulator()
    
    # Clean simulation
    job_clean = backend.run(circuit_clean, shots=1024)
    counts_clean = job_clean.result().get_counts(circuit_clean)
    
    # Eve simulation
    job_eve = backend.run(circuit_eve, shots=1024)
    counts_eve = job_eve.result().get_counts(circuit_eve)
    
    # Generate circuit images
    circuit_clean_img = fig_to_base64(circuit_clean.draw(output='mpl', style='iqp'))
    circuit_eve_img = fig_to_base64(circuit_eve.draw(output='mpl', style='iqp'))
    
    # Prepare histogram data
    histogram_data = []
    for state in ['00', '01', '10', '11']:
        histogram_data.append({
            "state": state,
            "without_eve": counts_clean.get(state[::-1], 0),  # Fix endianness
            "with_eve": counts_eve.get(state[::-1], 0)
        })
    
    return {
        "original_message": message,
        "encrypted_message": encrypted_message,
        "circuit_clean_image": circuit_clean_img,
        "circuit_eve_image": circuit_eve_img,
        "histogram_data": histogram_data,
        "counts_clean": counts_clean,
        "counts_eve": counts_eve
    }

# =============================================================================
#  FULL END-TO-END SIMULATION
# =============================================================================
def simulate_full_satellite_communication(message, num_qubits=50, has_eve=False):
    """
    Simulate complete satellite-ground communication
    """
    print(f"--- Running Full Satellite Communication Simulation ---")
    
    # Step 1: QKD Key Generation
    qkd_result = simulate_e91_qkd(num_qubits, has_eve)
    
    # Step 2: Message Encryption
    if len(qkd_result["key"]) < 2:
        return {"error": "Generated QKD key too short"}
    
    message_bits = [int(b) for b in message]
    key_bits = [int(b) for b in qkd_result["key"][:len(message)]]
    encrypted_bits = [(message_bits[i] ^ key_bits[i]) for i in range(len(message_bits))]
    encrypted_message = ''.join(map(str, encrypted_bits))
    
    # Step 3: Superdense Coding Transmission
    superdense_result = simulate_superdense_with_qkd(message, qkd_result["key"], has_eve)
    
    # Step 4: Decryption
    received_bits = encrypted_bits  # In real scenario, this would be received from transmission
    decrypted_bits = [(received_bits[i] ^ key_bits[i]) for i in range(len(received_bits))]
    decrypted_message = ''.join(map(str, decrypted_bits))
    
    # Step 5: Success analysis
    transmission_success = (message == decrypted_message)
    
    return {
        "qkd_result": qkd_result,
        "original_message": message,
        "encrypted_message": encrypted_message,
        "decrypted_message": decrypted_message,
        "transmission_success": transmission_success,
        "superdense_result": superdense_result,
        "steps": [
            {
                "name": "QKD Key Generation",
                "description": f"Generated {len(qkd_result['key'])}-bit key with {qkd_result['qber_percentage']:.1f}% QBER",
                "status": "completed"
            },
            {
                "name": "Message Encryption",
                "description": f"XOR encryption: {message} → {encrypted_message}",
                "status": "completed"
            },
            {
                "name": "Satellite Transmission",
                "description": f"Superdense coding transmission {'with' if has_eve else 'without'} Eve",
                "status": "completed"
            },
            {
                "name": "Message Decryption",
                "description": f"Decrypted: {decrypted_message} ({'Success' if transmission_success else 'Failed'})",
                "status": "completed" if transmission_success else "failed"
            }
        ]
    }

# =============================================================================
#  API ENDPOINTS
# =============================================================================

@app.route('/api/qkd_simulation', methods=['POST'])
def qkd_simulation_endpoint():
    """QKD E91 Protocol Simulation Endpoint"""
    try:
        data = request.get_json()
        num_pairs = data.get('num_pairs', 50)
        has_eve = data.get('has_eve', False)
        
        if num_pairs not in [10, 50, 100]:
            return jsonify({"error": "Invalid number of qubit pairs"}), 400
        
        result = simulate_e91_qkd(num_pairs, has_eve)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"QKD simulation error: {str(e)}"}), 500

@app.route('/api/superdense_qkd', methods=['POST'])
def superdense_qkd_endpoint():
    """Superdense Coding with QKD Keys Endpoint"""
    try:
        data = request.get_json()
        message = data.get('message')
        qkd_key = data.get('qkd_key', '')
        has_eve = data.get('has_eve', False)
        
        if not message or message not in ['00', '01', '10', '11']:
            return jsonify({"error": "Invalid message provided"}), 400
        
        result = simulate_superdense_with_qkd(message, qkd_key, has_eve)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Superdense coding error: {str(e)}"}), 500

@app.route('/api/full_simulation', methods=['POST'])
def full_simulation_endpoint():
    """Full Satellite Communication Simulation Endpoint"""
    try:
        data = request.get_json()
        message = data.get('message')
        num_qubits = data.get('num_qubits', 50)
        has_eve = data.get('has_eve', False)
        
        if not message or message not in ['00', '01', '10', '11']:
            return jsonify({"error": "Invalid message provided"}), 400
        
        if num_qubits not in [10, 50, 100]:
            return jsonify({"error": "Invalid number of qubits"}), 400
        
        result = simulate_full_satellite_communication(message, num_qubits, has_eve)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Full simulation error: {str(e)}"}), 500

@app.route('/api/analysis_data', methods=['GET'])
def analysis_data_endpoint():
    """Generate analysis data for charts"""
    try:
        # QBER vs Number of qubit pairs
        qber_data = []
        for num_pairs in [10, 20, 30, 40, 50, 75, 100]:
            # Without Eve
            result_clean = simulate_e91_qkd(num_pairs, False)
            # With Eve  
            result_eve = simulate_e91_qkd(num_pairs, True)
            
            qber_data.append({
                "num_pairs": num_pairs,
                "without_eve": result_clean["qber_percentage"],
                "with_eve": result_eve["qber_percentage"]
            })
        
        # Success rate data
        success_data = []
        for num_pairs in [10, 50, 100]:
            successes_clean = 0
            successes_eve = 0
            trials = 10
            
            for _ in range(trials):
                test_message = random.choice(['00', '01', '10', '11'])
                
                # Without Eve
                result_clean = simulate_full_satellite_communication(test_message, num_pairs, False)
                if result_clean.get("transmission_success"):
                    successes_clean += 1
                
                # With Eve
                result_eve = simulate_full_satellite_communication(test_message, num_pairs, True)
                if result_eve.get("transmission_success"):
                    successes_eve += 1
            
            success_data.append({
                "num_pairs": num_pairs,
                "without_eve": (successes_clean / trials) * 100,
                "with_eve": (successes_eve / trials) * 100
            })
        
        return jsonify({
            "qber_data": qber_data,
            "success_data": success_data
        })
        
    except Exception as e:
        return jsonify({"error": f"Analysis data error: {str(e)}"}), 500

# =============================================================================
#  RUN THE APP
# =============================================================================
if __name__ == '__main__':
    print("Starting Satellite Communication Flask server...")
    app.run(debug=True, port=5001)