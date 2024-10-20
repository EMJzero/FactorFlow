# This script imports and wraps Accelergy to query directly its plug-ins like CACTI and NeuroSim.
# Given the overhead of Accelergy on the overall runtime, import this script only when truly needed.

# Install Accelergy from: https://timeloop.csail.mit.edu/v4/installation

from functools import partial
import importlib
import logging
import sys
import os

if __name__ == "__main__":
    sys.path.append("..")
from settings import *

print("------ loading Accelergy ------")

# Prepare Accelergy plug-ins
try:
    import accelergy.raw_inputs_2_dicts as accelergy_raw_inputs_2_dicts # type: ignore
    import accelergy.system_state as accelergy_system_state # type: ignore
    import accelergy.plug_in_path_to_obj as accelergy_plug_in_path_to_obj # type: ignore
    from accelergy.plug_in_interface.query_plug_ins import get_best_estimate as accelergy_get_best_estimate # type: ignore
except:
    print("WARNING: Accelergy package not found or incompatible, trying a manual import from ACCELERGY_PATH.")
    assert os.path.exists(Settings.ACCELERGY_PATH), f"The provided ACCELERGY_PATH ({Settings.ACCELERGY_PATH}) does not exist."
    sys.path.append(Settings.ACCELERGY_PATH)
    
    assert os.path.exists(Settings.ACCELERGY_PATH + "/accelergy/raw_inputs_2_dicts.py"), f"Invalid ACCELERGY_PATH ({Settings.ACCELERGY_PATH}) or unsupported Accelergy version. FactorFlow could not find 'ACCELERGY_PATH/accelergy/raw_inputs_2_dicts.py'."
    accelergy_raw_inputs_2_dicts = importlib.import_module('accelergy.raw_inputs_2_dicts')
    assert os.path.exists(Settings.ACCELERGY_PATH + "/accelergy/system_state.py"), f"Invalid ACCELERGY_PATH ({Settings.ACCELERGY_PATH}) or unsupported Accelergy version. FactorFlow could not find 'ACCELERGY_PATH/accelergy/system_state.py'."
    accelergy_system_state = importlib.import_module('accelergy.system_state')
    assert os.path.exists(Settings.ACCELERGY_PATH + "/accelergy/plug_in_path_to_obj.py"), f"Invalid ACCELERGY_PATH ({Settings.ACCELERGY_PATH}) or unsupported Accelergy version. FactorFlow could not find 'ACCELERGY_PATH/accelergy/plug_in_path_to_obj.py'."
    accelergy_plug_in_path_to_obj = importlib.import_module('accelergy.plug_in_path_to_obj')
    assert os.path.exists(Settings.ACCELERGY_PATH + "/accelergy/plug_in_interface/query_plug_ins.py"), f"Invalid ACCELERGY_PATH ({Settings.ACCELERGY_PATH}) or unsupported Accelergy version. FactorFlow could not find 'ACCELERGY_PATH/accelergy/plug_in_interface/query_plug_ins.py'."
    #from accelergy.plug_in_interface.query_plug_ins import get_best_estimate
    accelergy_query_plug_ins = importlib.import_module('accelergy.plug_in_interface.query_plug_ins')
    accelergy_get_best_estimate = getattr(accelergy_query_plug_ins, 'get_best_estimate')

# Reduce the logging level for Accelergy, default was logging.INFO (alternatives: WARN, ERROR, DEBUG)
logging.getLogger().setLevel(logging.WARN)

# Prepare Accelergy plug-ins
accelergy_raw_dicts = accelergy_raw_inputs_2_dicts.RawInputs2Dicts({'path_arglist': [], 'parser_version': '0.4'}, False)
accelergy_state = accelergy_system_state.SystemState()
#print("PLUG-IN PATHS:", accelergy_raw_dicts.get_estimation_plug_in_paths(), accelergy_raw_dicts.get_python_plug_in_paths())
accelergy_state.add_plug_ins(
            accelergy_plug_in_path_to_obj.plug_in_path_to_obj(
                accelergy_raw_dicts.get_estimation_plug_in_paths(),
                accelergy_raw_dicts.get_python_plug_in_paths(),
                'tmp',
            ),
        )
print("ACCELERGY PLUG-INS:", list(map(lambda p : p.get_name(), accelergy_state.plug_ins)))

# Prepare the query function
# NOTE: for details on how queries are handled, read the code in .../accelergy/plug_in_interface/query_plug_ins.py lines 116-209
# NOTE: for details on queries syntax, read the code in .../accelergy/plug_in_interface/interface.py lines 219-243
# NOTE: results are an instance of Estimation (.../accelergy/plug_in_interface/interface.py lines 99-161), use .get_value() for results in Joules, then scale them to PicoJoules.
# NOTE: precision defaults to 6 (decimal places returned)
# NOTE: actions are 'read', 'write', 'update', 'leak' (only 'read' and 'leak' supported by non-memory components) ('compare' is exclusive to the comparator)
accelergy_estimate_energy_raw = partial(accelergy_get_best_estimate, plug_ins=accelergy_state.plug_ins, is_energy_estimation=True)
accelergy_estimate_energy = lambda query : accelergy_estimate_energy_raw(query=query).get_value()*(10**12)

if __name__ == "__main__":
    print("\nTesting Accelergy:")
    print("LPDDR4 estimation test:", accelergy_estimate_energy(query={
        "class_name": "DRAM",
        "attributes": {
            "type": "LPDDR4",
            "width": 64,
            #"datawidth": 8,
            #"has_power_gating": False,
            #"n_banks": 2,
            #"cluster_size": 1,
            #"reduction_supported": True,
            #"multiple_buffering": 1,
            #"allow_overbooking": False,
            #"global_cycle_seconds": 1.2e-09,
            "technology": "32nm",
            #"action_latency_cycles": 1,
            "cycle_seconds": 1.2e-09,
            #"n_instances": 1
        },
        "action_name": "read",
        "arguments": {
            #"global_cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1
        }
    }), "pJ")
    
    print("SRAM estimation test:", accelergy_estimate_energy(query={
        "class_name": "SRAM",
        "attributes": {
            "n_rw_ports": 1,
            #"n_rdwr_ports": 1,
            #"n_rd_ports": 0,
            #"n_wr_ports": 0,
            "width": 64,
            "depth": 16384,
            #"global_cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1,
            #"cycle_seconds": 1.2e-09,
            #"n_instances": 1,
            "technology": "32nm",
            "n_banks": 1,
            #"latency": "5ns"
        },
        "action_name": "read",
        "arguments": {
            #"global_cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1
            #"cycle_seconds": 1.2e-09,
            #"n_instances": 1,
            #"technology": "32nm",
        }
    }), "pJ")
    
    # bank of 192 registers
    dynamic_energy_scale = (16/32)*((12/64)**(1.56/2)) # taken from smartbuffer_RF
    print("Register estimation test:", accelergy_estimate_energy(query={
        "class_name": "aladdin_register",
        "attributes": {
            #"global_cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1,
            #"cycle_seconds": 1.2e-09,
            #"n_instances": 1,
            "technology": "32nm",
            #"n_banks": 1,
            #"latency": "5ns"
        },
        "action_name": "read",
        "arguments": {
            #"global_cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1
            #"cycle_seconds": 1.2e-09,
            #"n_instances": 1,
            #"technology": "32nm",
        }
    })*32 + accelergy_estimate_energy(query={ # the 32 comes from smartbuffer_RF with max(32, width)
        "class_name": "aladdin_comparator",
        "attributes": {
            #"global_cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1,
            #"cycle_seconds": 1.2e-09,
            #"n_instances": 1,
            "technology": "32nm",
            #"n_banks": 1,
            #"latency": "5ns"
        },
        "action_name": "compare",
        "arguments": {
            #"global_cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1
            #"cycle_seconds": 1.2e-09,
            #"n_instances": 1,
            #"technology": "32nm",
        }
    })*64 + accelergy_estimate_energy(query={ # the 64 comes from smartbuffer_RF with max(64, depth)
        "class_name": "intadder",
        "attributes": {
            "n_bits": 8,
            "precision": 8,
            "datawidth": 8,
            #"n_instances": 1,
            "technology": "32nm",
            "global_cycle_seconds": 1.2e-09,
            "cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1
            #"latency": "5ns"
        },
        "action_name": "add",
        "arguments": {
            #"global_cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1
            #"cycle_seconds": 1.2e-09,
            #"n_instances": 1,
            #"technology": "32nm",
        }
    })*1, "pJ") # these are the 2 (whops, only 1 used while reading) address generators

    print("MAC estimation test:", accelergy_estimate_energy(query={
        "class_name": "aladdin_adder",
        "attributes": {
            # output bitwidth
            "width": 16,
            #"global_cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1,
            #"cycle_seconds": 1.2e-09,
            "technology": "32nm",
            #"n_instances": 1,
        },
        "action_name": "read",
        "arguments": {
            #"global_cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1,
            #"cycle_seconds": 1.2e-09,
            #"technology": "32nm",
            #"n_instances": 1,
        }
    }) + accelergy_estimate_energy(query={
        "class_name": "aladdin_adder",
        "attributes": {
            # the two operands bitwidths
            "width_a": 8,
            "width_b": 8,
            #"global_cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1,
            #"cycle_seconds": 1.2e-09,
            "technology": "32nm",
            #"n_instances": 1,
        },
        "action_name": "read",
        "arguments": {
            #"global_cycle_seconds": 1.2e-09,
            #"action_latency_cycles": 1,
            #"cycle_seconds": 1.2e-09,
            #"technology": "32nm",
            #"n_instances": 1,
        }
    }), "pJ")
