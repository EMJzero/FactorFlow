# This script imports and wraps Accelergy to query directly its plug-ins like CACTI and NeuroSim.
# Given the overhead of Accelergy on the overall runtime, import this script only when truly needed.

from functools import partial
import importlib
import logging
import sys
import os

from settings import *

print("------ loading Accelergy ------")

# Prepare Accelergy plug-ins
try:
    import accelergy.raw_inputs_2_dicts as accelergy_raw_inputs_2_dicts
    import accelergy.system_state as accelergy_system_state
    import accelergy.plug_in_path_to_obj as accelergy_plug_in_path_to_obj
    from accelergy.plug_in_interface.query_plug_ins import get_best_estimate as accelergy_get_best_estimate
except:
    print("WARNING: Accelergy package not found or incompatible, trying a manual import from ACCELERGY_PATH.")
    assert os.path.exists(Settings.ACCELERGY_PATH), f"The provided ACCELERGY_PATH ({Settings.ACCELERGY_PATH}) does not exist."
    sys.path.append(Settings.ACCELERGY_PATH)
    
    assert os.path.exists(Settings.ACCELERGY_PATH + "/accelergy/raw_inputs_2_dicts.py"), f"Invalid ACCELERGY_PATH ({Settings.ACCELERGY_PATH}) or unsupported Accelergy version. FactorFlow could not find 'ACCELERGY_PATH/accelergy/raw_inputs_2_dicts.py'."
    accelergy_raw_inputs_2_dicts = importlib.import_module('accelergy.raw_inputs_2_dicts')
    accelergy_raw_dicts = accelergy_raw_inputs_2_dicts.RawInputs2Dicts({'path_arglist': [], 'parser_version': '0.4'}, False)
    assert os.path.exists(Settings.ACCELERGY_PATH + "/accelergy/system_state.py"), f"Invalid ACCELERGY_PATH ({Settings.ACCELERGY_PATH}) or unsupported Accelergy version. FactorFlow could not find 'ACCELERGY_PATH/accelergy/system_state.py'."
    accelergy_system_state = importlib.import_module('accelergy.system_state')
    assert os.path.exists(Settings.ACCELERGY_PATH + "/accelergy/plug_in_path_to_obj.py"), f"Invalid ACCELERGY_PATH ({Settings.ACCELERGY_PATH}) or unsupported Accelergy version. FactorFlow could not find 'ACCELERGY_PATH/accelergy/plug_in_path_to_obj.py'."
    accelergy_plug_in_path_to_obj = importlib.import_module('accelergy.plug_in_path_to_obj')
    assert os.path.exists(Settings.ACCELERGY_PATH + "/accelergy/plug_in_interface/query_plug_ins.py"), f"Invalid ACCELERGY_PATH ({Settings.ACCELERGY_PATH}) or unsupported Accelergy version. FactorFlow could not find 'ACCELERGY_PATH/accelergy/plug_in_interface/query_plug_ins.py'."
    #from accelergy.plug_in_interface.query_plug_ins import get_best_estimate
    accelergy_query_plug_ins = importlib.import_module('accelergy.plug_in_interface.query_plug_ins')

# reduce the logging level for Accelergy, default was logging.INFO
logging.getLogger().setLevel(logging.WARN)

accelergy_state = accelergy_system_state.SystemState()
#print("PLUG-IN PATHS:", accelergy_raw_dicts.get_estimation_plug_in_paths(), accelergy_raw_dicts.get_python_plug_in_paths())
accelergy_state.add_plug_ins(
            accelergy_plug_in_path_to_obj.plug_in_path_to_obj(
                accelergy_raw_dicts.get_estimation_plug_in_paths(),
                accelergy_raw_dicts.get_python_plug_in_paths(),
                'tmp',
            ),
        )

# Prepare the query function
accelergy_get_best_estimate = getattr(accelergy_query_plug_ins, 'get_best_estimate')
print("ACCELERGY PLUG-INS:", list(map(lambda p : p.get_name(), accelergy_state.plug_ins)))

# NOTE: for details on how queries are handled, read the code in .../accelergy/plug_in_interface/query_plug_ins.py lines 116-209
# NOTE: for details on queries syntax, read the code in .../accelergy/plug_in_interface/interface.py lines 219-243
# NOTE: precision defaults to 6 (decimal places returned)
accelergy_estimate_energy = partial(accelergy_get_best_estimate, plug_ins=accelergy_state.plug_ins, is_energy_estimation=True)


if __name__ == "__main__":
    print("LPDDR4 estimation test:", accelergy_estimate_energy(query={
        "class_name": "DRAM",
        "attributes": {
            "type": "LPDDR4",
            "width": 64,
            "datawidth": 8,
            "has_power_gating": False,
            "n_banks": 2,
            "cluster_size": 1,
            "reduction_supported": True,
            "multiple_buffering": 1,
            "allow_overbooking": False,
            "global_cycle_seconds": 1.2e-09,
            "technology": "32nm",
            "action_latency_cycles": 1,
            "cycle_seconds": 1.2e-09,
            "n_instances": 1
        },
        "action_name": "read",
        "arguments": {
            "global_cycle_seconds": 1.2e-09,
            "action_latency_cycles": 1
        }
    }), "J")