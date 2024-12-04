from typing import Union, Any

from architectures.solutions_db import *
from settings import *
from factors import *
from engine import *
from utils import *
from arch import *

"""
Runs a test by recomputing an architecture's statistics given a mapping in
the form of a complete constraints set. The test passes iif all the newly
evaluated statistics match those in "correct_data".
"""
def runTest(arch : Arch, correct_data : list[dict[str, Union[int, float]]], comp : Shape, bias_read : bool, change_settings : dict[str, Any] = None) -> None:
    if change_settings:
        backup = {key: getattr(Settings, key) for key in change_settings.keys()}
        for key, value in change_settings.items():
            setattr(Settings, key, value)
    arch.initFactors(comp)
    arch.enforceFactorsConstraints()
    assert arch.checkDataflowConstraints(), "Dataflow constraints violated."
    updateStats(arch, bias_read)
    mem = list(filter(lambda l : isinstance(l, MemLevel), arch))
    passed = True
    for i in range(len(mem)):
        level = mem[i]
        correct_data_level = correct_data[i]
        for key in correct_data_level:
            if not level[key] == correct_data_level[key]:
                passed = False
                print(f"{level.name}:{chr(9) * (2 - (len(level.name) + 1)//8)}Error on result: {key}\n\t\tExpected: {correct_data_level[key]}\n\t\tObtained: {level[key]}")
    if change_settings:
        for key, value in backup.items():
            setattr(Settings, key, value)
    assert passed, "Test FAILED..."

"""
Helper function that dumps the entirety of an architecture's MOPs statistics
in the form of python code for an array of dictionaries, the same as those
below in this file.
"""
def generateTestMOPs(arch : Arch) -> None:
    string = "correct_mops = ["
    indentation = 1
    for level in arch:
        if isinstance(level, MemLevel):
            string += "{\n"
            indentation += 1
            for key in ["in_reads", "w_reads", "out_reads", "in_writes", "w_writes", "out_writes", "last_out_reads", "last_out_writes"]:
                string += f"{'    '*indentation}\"{key}\": {level[key]},\n"
            indentation -= 1
            string = string[:-2] + f"\n{'    '*indentation}" + "},"
        elif isinstance(level, FanoutLevel):
            continue
        elif isinstance(level, ComputeLevel):
            break
    string = string[:-1] + "]\n"
    print(string)

"""
Helper function that dumps the entirety of an architecture's latency statistics
in the form of python code for an array of dictionaries, the same as those
below in this file.
"""
def generateTestLatency(arch : Arch) -> None:
    string = "correct_latency = ["
    indentation = 1
    for level in arch:
        if isinstance(level, MemLevel):
            string += "{\n"
            indentation += 1
            for key in ["latency_read_drain", "latency_fill_update", "cc_per_tile", "stall_cycles", "ideal_bandwidth_read", "ideal_bandwidth_update", "ideal_bandwidth_fill", "ideal_bandwidth_drain"]:
                string += f"{'    '*indentation}\"{key}\": {level[key]},\n"
            indentation -= 1
            string = string[:-2] + f"\n{'    '*indentation}" + "},"
        elif isinstance(level, FanoutLevel):
            continue
        elif isinstance(level, ComputeLevel):
            break
    string = string[:-1] + "]\n"
    print(string)

correct_mops_gemmini_timeloop = [{
        "in_reads": 4194304,
        "w_reads": 50331648,
        "out_reads": 25165824,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 25165824,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 805306368,
        "w_reads": 50331648,
        "out_reads": 0,
        "in_writes": 4194304,
        "w_writes": 50331648,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 830472192,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 830472192,
        "last_out_reads": 25165824,
        "last_out_writes": 25165824
    },{
        "in_reads": 0,
        "w_reads": 12884901888,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 50331648,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    }]

correct_latency_gemmini_timeloop = [{
        "latency_read_drain": 53477376.0,
        "latency_fill_update": 53477376.0,
        "cc_per_tile": 26112.0,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 1.4901960784313726,
        "ideal_bandwidth_update": 0.47058823529411764,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 53477376.0,
        "latency_fill_update": 50331648,
        "cc_per_tile": 3072,
        "stall_cycles": 3145728.0,
        "ideal_bandwidth_read": 17.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 1.0833333333333333,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 50331648,
        "latency_fill_update": 50331648,
        "cc_per_tile": 16,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.03125,
        "ideal_bandwidth_drain": 0.03125
    },{
        "latency_read_drain": 50331648,
        "latency_fill_update": 50331648,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.00390625,
        "ideal_bandwidth_drain": 0.0
    }]

correct_mops_gemmini_factorflow_1 = [{
        "in_reads": 4194304,
        "w_reads": 25165824,
        "out_reads": 25165824,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 25165824,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 805306368,
        "w_reads": 805306368,
        "out_reads": 0,
        "in_writes": 4194304,
        "w_writes": 25165824,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 830472192,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 830472192,
        "last_out_reads": 25165824,
        "last_out_writes": 25165824
    },{
        "in_reads": 0,
        "w_reads": 12884901888,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 805306368,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    }]

correct_latency_gemmini_factorflow_1 = [{
        "latency_read_drain": 100663296.0,
        "latency_fill_update": 100663296.0,
        "cc_per_tile": 1048576.0,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 0.5416666666666666,
        "ideal_bandwidth_update": 0.25,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 100663296.0,
        "latency_fill_update": 50331648,
        "cc_per_tile": 512,
        "stall_cycles": 50331648.0,
        "ideal_bandwidth_read": 32.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.5833333333333334,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 50331648,
        "latency_fill_update": 50331648,
        "cc_per_tile": 16,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.03125,
        "ideal_bandwidth_drain": 0.03125
    },{
        "latency_read_drain": 50331648,
        "latency_fill_update": 50331648,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.0625,
        "ideal_bandwidth_drain": 0.0
    }]

correct_mops_gemmini_factorflow_2 = [{
        "in_reads": 4194304,
        "w_reads": 3145728,
        "out_reads": 188743680,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 201326592,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 805306368,
        "w_reads": 3145728,
        "out_reads": 0,
        "in_writes": 4194304,
        "w_writes": 3145728,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 994050048,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 994050048,
        "last_out_reads": 188743680,
        "last_out_writes": 201326592
    },{
        "in_reads": 0,
        "w_reads": 12884901888,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 3145728,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    }]

correct_latency_gemmini_factorflow_2 = [{
        "latency_read_drain": 50528256.0,
        "latency_fill_update": 50528256.0,
        "cc_per_tile": 526336.0,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 3.880674448767834,
        "ideal_bandwidth_update": 3.9844357976653697,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 50528256.0,
        "latency_fill_update": 50331648,
        "cc_per_tile": 4096,
        "stall_cycles": 196608.0,
        "ideal_bandwidth_read": 16.0625,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.14583333333333334,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 50331648,
        "latency_fill_update": 50331648,
        "cc_per_tile": 16,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.984375,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.234375,
        "ideal_bandwidth_drain": 0.25
    },{
        "latency_read_drain": 50331648,
        "latency_fill_update": 50331648,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.000244140625,
        "ideal_bandwidth_drain": 0.0
    }]

correct_mops_eyeriss_timeloop = [{
        "in_reads": 67108864,
        "w_reads": 50331648,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 12582912,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 67108864,
        "w_reads": 0,
        "out_reads": 805306368,
        "in_writes": 67108864,
        "w_writes": 0,
        "out_writes": 805306368,
        "last_out_reads": 0,
        "last_out_writes": 12582912
    },{
        "in_reads": 12884901888,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 6442450944,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 12884901888,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 50331648,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 13677625344,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 13677625344,
        "last_out_reads": 792723456,
        "last_out_writes": 805306368
    }]

correct_latency_eyeriss_timeloop = [{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 8192,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.875,
        "ideal_bandwidth_update": 0.09375,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 32,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 6.40625,
        "ideal_bandwidth_update": 6.0,
        "ideal_bandwidth_fill": 0.5,
        "ideal_bandwidth_drain": 0.09375
    },{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 32,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.5,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 2,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.00390625,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9990234375,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.0615234375,
        "ideal_bandwidth_drain": 0.0625
    }]

correct_mops_eyeriss_factorflow_1 = [{
        "in_reads": 33554432,
        "w_reads": 50331648,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 12582912,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 33554432,
        "w_reads": 0,
        "out_reads": 805306368,
        "in_writes": 33554432,
        "w_writes": 0,
        "out_writes": 805306368,
        "last_out_reads": 0,
        "last_out_writes": 12582912
    },{
        "in_reads": 12884901888,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 3221225472,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 12884901888,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 50331648,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 13677625344,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 13677625344,
        "last_out_reads": 792723456,
        "last_out_writes": 805306368
    }]

correct_latency_eyeriss_factorflow_1 = [{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 131072,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.625,
        "ideal_bandwidth_update": 0.09375,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 64,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 6.15625,
        "ideal_bandwidth_update": 6.0,
        "ideal_bandwidth_fill": 0.25,
        "ideal_bandwidth_drain": 0.09375
    },{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 64,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.25,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 4,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.00390625,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9990234375,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.0615234375,
        "ideal_bandwidth_drain": 0.0625
    }]

correct_mops_simba_timeloop = [{
        "in_reads": 4194304,
        "w_reads": 402653184,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 12582912,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 12582912,
        "w_reads": 0,
        "out_reads": 402653184,
        "in_writes": 4194304,
        "w_writes": 0,
        "out_writes": 402653184,
        "last_out_reads": 0,
        "last_out_writes": 12582912
    },{
        "in_reads": 6442450944,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 25165824,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 3221225472,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 402653184,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 4001366016,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 4001366016,
        "last_out_reads": 780140544,
        "last_out_writes": 805306368
    },{
        "in_reads": 0,
        "w_reads": 12884901888,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 3221225472,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    }]

correct_latency_simba_timeloop = [{
        "latency_read_drain": 402653184,
        "latency_fill_update": 402653184,
        "cc_per_tile": 1048576,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0104166666666667,
        "ideal_bandwidth_update": 0.03125,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 402653184,
        "latency_fill_update": 402653184,
        "cc_per_tile": 4096,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.010416666666666666,
        "ideal_bandwidth_drain": 0.03125
    },{
        "latency_read_drain": 402653184,
        "latency_fill_update": 402653184,
        "cc_per_tile": 2048,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 4.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.015625,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 402653184,
        "latency_fill_update": 402653184,
        "cc_per_tile": 8,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.125,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 402653184,
        "latency_fill_update": 402653184,
        "cc_per_tile": 2,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9921875,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.2421875,
        "ideal_bandwidth_drain": 0.25
    },{
        "latency_read_drain": 402653184,
        "latency_fill_update": 402653184,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.25,
        "ideal_bandwidth_drain": 0.0
    }]

correct_mops_simba_factorflow_1 = [{
        "in_reads": 8388608.0,
        "w_reads": 3145728.0,
        "out_reads": 0.0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 12582912.0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 8388608.0,
        "w_reads": 0.0,
        "out_reads": 12582912.0,
        "in_writes": 8388608.0,
        "w_writes": 0,
        "out_writes": 12582912.0,
        "last_out_reads": 0.0,
        "last_out_writes": 12582912.0
    },{
        "in_reads": 3221225472.0,
        "w_reads": 0.0,
        "out_reads": 0.0,
        "in_writes": 67108864.0,
        "w_writes": 0,
        "out_writes": 0.0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0.0,
        "w_reads": 805306368.0,
        "out_reads": 0.0,
        "in_writes": 0,
        "w_writes": 3145728.0,
        "out_writes": 0.0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0.0,
        "w_reads": 0.0,
        "out_reads": 3221225472.0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 3221225472.0,
        "last_out_reads": 0.0,
        "last_out_writes": 25165824.0
    },{
        "in_reads": 0,
        "w_reads": 12884901888,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 805306368.0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    }]

correct_latency_simba_factorflow_1 = [{
        "latency_read_drain": 50331648,
        "latency_fill_update": 50331648,
        "cc_per_tile": 98304,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.22916666666666666,
        "ideal_bandwidth_update": 0.25,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 50331648,
        "latency_fill_update": 50331648,
        "cc_per_tile": 98304,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.16666666666666666,
        "ideal_bandwidth_update": 0.25,
        "ideal_bandwidth_fill": 0.16666666666666666,
        "ideal_bandwidth_drain": 0.25
    },{
        "latency_read_drain": 50331648,
        "latency_fill_update": 50331648,
        "cc_per_tile": 98304,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 4.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.08333333333333333,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 50331648,
        "latency_fill_update": 50331648,
        "cc_per_tile": 16384,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.25,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.0009765625,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 50331648,
        "latency_fill_update": 50331648,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9921875,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0078125
    },{
        "latency_read_drain": 50331648,
        "latency_fill_update": 50331648,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.0625,
        "ideal_bandwidth_drain": 0.0
    }]

correct_mops_tpu_factorflow_1 = [{
        "in_reads": 16777216.0,
        "w_reads": 0.0,
        "out_reads": 0.0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 262144.0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0.0,
        "w_reads": 262144.0,
        "out_reads": 0.0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 0.0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 16777216.0,
        "w_reads": 0.0,
        "out_reads": 0.0,
        "in_writes": 16777216.0,
        "w_writes": 0,
        "out_writes": 0.0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0.0,
        "w_reads": 8388608.0,
        "out_reads": 0.0,
        "in_writes": 0,
        "w_writes": 262144.0,
        "out_writes": 0.0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0.0,
        "w_reads": 0.0,
        "out_reads": 4194304.0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 4194304.0,
        "last_out_reads": 0.0,
        "last_out_writes": 262144.0
    },{
        "in_reads": 0,
        "w_reads": 1073741824,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 8388608.0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    }]

correct_latency_tpu_factorflow_1 = [{
        "latency_read_drain": 4194304.0,
        "latency_fill_update": 2097152.0,
        "cc_per_tile": 2097152.0,
        "stall_cycles": 2097152.0,
        "ideal_bandwidth_read": 8.0,
        "ideal_bandwidth_update": 0.125,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 2097152.0,
        "latency_fill_update": 2097152.0,
        "cc_per_tile": 65536.0,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 0.125,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 2097152.0,
        "latency_fill_update": 2097152.0,
        "cc_per_tile": 65536.0,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 8.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 8.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 2097152.0,
        "latency_fill_update": 65536,
        "cc_per_tile": 2048,
        "stall_cycles": 2031616.0,
        "ideal_bandwidth_read": 128.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 4.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 65536,
        "latency_fill_update": 65536,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9375,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0625
    },{
        "latency_read_drain": 65536,
        "latency_fill_update": 65536,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.0078125,
        "ideal_bandwidth_drain": 0.0
    }]

correct_mops_tpu_timeloop_ex = [{
        "in_reads": 65536,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 131072,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 131072,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 131072,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 65536,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 131072,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 131072,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 131072,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 131072,
        "last_out_reads": 0,
        "last_out_writes": 131072
    },{
        "in_reads": 0,
        "w_reads": 33554432,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 131072,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    }]

correct_latency_tpu_timeloop_ex = [{
        "latency_read_drain": 32768.0,
        "latency_fill_update": 32768.0,
        "cc_per_tile": 32768.0,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 2.0,
        "ideal_bandwidth_update": 4.0,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 32768.0,
        "latency_fill_update": 32768.0,
        "cc_per_tile": 32768.0,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 4.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 32768.0,
        "latency_fill_update": 32768.0,
        "cc_per_tile": 32768.0,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 4.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 2.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 32768.0,
        "latency_fill_update": 32768.0,
        "cc_per_tile": 512,
        "stall_cycles": 32256.0,
        "ideal_bandwidth_read": 256.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 256.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 512,
        "latency_fill_update": 512,
        "cc_per_tile": 256,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.0,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 1.0
    },{
        "latency_read_drain": 512,
        "latency_fill_update": 512,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.00390625,
        "ideal_bandwidth_drain": 0.0
    }]

correct_mops_eyeriss_timeloop_ex_1 = [{
        "in_reads": 8388608,
        "w_reads": 131072,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 65536,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 8388608,
        "w_reads": 0,
        "out_reads": 524288,
        "in_writes": 8388608,
        "w_writes": 0,
        "out_writes": 524288,
        "last_out_reads": 0,
        "last_out_writes": 65536
    },{
        "in_reads": 67108864,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 67108864,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 67108864,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 131072,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 70778880,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 70778880,
        "last_out_reads": 3670016,
        "last_out_writes": 4194304
    }]

correct_latency_eyeriss_timeloop_ex_1 = [{
        "latency_read_drain": 2129920.0,
        "latency_fill_update": 1048576,
        "cc_per_tile": 8192,
        "stall_cycles": 1081344.0,
        "ideal_bandwidth_read": 8.125,
        "ideal_bandwidth_update": 0.0625,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 1048576,
        "latency_fill_update": 1048576,
        "cc_per_tile": 16,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 8.4375,
        "ideal_bandwidth_update": 0.5,
        "ideal_bandwidth_fill": 8.0,
        "ideal_bandwidth_drain": 0.0625
    },{
        "latency_read_drain": 1048576,
        "latency_fill_update": 1048576,
        "cc_per_tile": 16,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 1.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 1048576,
        "latency_fill_update": 1048576,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.001953125,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 1048576,
        "latency_fill_update": 1048576,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9921875,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.0546875,
        "ideal_bandwidth_drain": 0.0625
    }]

correct_mops_eyeriss_timeloop_ex_2 = [{
        "in_reads": 33554432,
        "w_reads": 50331648,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 12582912,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 67108864,
        "w_reads": 0,
        "out_reads": 100663296,
        "in_writes": 33554432,
        "w_writes": 0,
        "out_writes": 100663296,
        "last_out_reads": 0,
        "last_out_writes": 12582912
    },{
        "in_reads": 12884901888,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 805306368,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 12884901888,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 50331648,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 13589544960,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 13589544960,
        "last_out_reads": 704643072,
        "last_out_writes": 805306368
    }]

correct_latency_eyeriss_timeloop_ex_2 = [{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 131072,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.625,
        "ideal_bandwidth_update": 0.09375,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 256,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.15625,
        "ideal_bandwidth_update": 0.75,
        "ideal_bandwidth_fill": 0.25,
        "ideal_bandwidth_drain": 0.09375
    },{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 256,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.0625,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 16,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.00390625,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 134217728,
        "latency_fill_update": 134217728,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9921875,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.0546875,
        "ideal_bandwidth_drain": 0.0625
    }]

correct_mops_eyeriss_conv_timeloop_1 = [{
        "in_reads": 1376256,
        "w_reads": 33030144,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 802816,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 57802752.0,
        "w_reads": 0,
        "out_reads": 154140672,
        "in_writes": 1376256,
        "w_writes": 0,
        "out_writes": 154140672,
        "last_out_reads": 0,
        "last_out_writes": 802816
    },{
        "in_reads": 1849688064.0,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 115605504.0,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 1849688064,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 462422016,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 2463039488,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 2463039488,
        "last_out_reads": 613351424,
        "last_out_writes": 616562688
    }]

correct_latency_eyeriss_conv_timeloop_1 = [{
        "latency_read_drain": 16515072,
        "latency_fill_update": 16515072,
        "cc_per_tile": 36864,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 2.0833333333333335,
        "ideal_bandwidth_update": 0.04861111111111111,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 16515072,
        "latency_fill_update": 16515072,
        "cc_per_tile": 24,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 12.784722222222221,
        "ideal_bandwidth_update": 9.333333333333334,
        "ideal_bandwidth_fill": 0.08333333333333333,
        "ideal_bandwidth_drain": 0.04861111111111111
    },{
        "latency_read_drain": 16515072,
        "latency_fill_update": 16515072,
        "cc_per_tile": 24,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.0625,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 16515072,
        "latency_fill_update": 16515072,
        "cc_per_tile": 8,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.25,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 16515072,
        "latency_fill_update": 16515072,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9982638888888888,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.3315972222222222,
        "ideal_bandwidth_drain": 0.3333333333333333
    }]

correct_mops_eyeriss_conv_timeloop_2 = [{
        "in_reads": 15204352,
        "w_reads": 18874368,
        "out_reads": 2408448,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 3211264,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 198180864.0,
        "w_reads": 0,
        "out_reads": 15253504,
        "in_writes": 15204352,
        "w_writes": 0,
        "out_writes": 15253504,
        "last_out_reads": 2408448,
        "last_out_writes": 3211264
    },{
        "in_reads": 1849688064.0,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 396361728.0,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 1849688064,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 264241152,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 1936392192,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 1921941504,
        "last_out_reads": 72253440,
        "last_out_writes": 91521024
    }]

correct_latency_eyeriss_conv_timeloop_2 = [{
        "latency_read_drain": 13339648.0,
        "latency_fill_update": 13339648.0,
        "cc_per_tile": 6513.5,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 2.735242189299148,
        "ideal_bandwidth_update": 0.24073078989790436,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 13339648.0,
        "latency_fill_update": 11010048,
        "cc_per_tile": 48,
        "stall_cycles": 2329600.0,
        "ideal_bandwidth_read": 19.09375,
        "ideal_bandwidth_update": 1.1666666666666667,
        "ideal_bandwidth_fill": 1.599702380952381,
        "ideal_bandwidth_drain": 0.2916666666666667
    },{
        "latency_read_drain": 11010048,
        "latency_fill_update": 11010048,
        "cc_per_tile": 48,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.2142857142857143,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 11010048,
        "latency_fill_update": 11010048,
        "cc_per_tile": 2,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.14285714285714285,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 11010048,
        "latency_fill_update": 11010048,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9973958333333334,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.0390625,
        "ideal_bandwidth_drain": 0.049479166666666664
    }]

correct_mops_eyeriss_conv_timeloop_3 = [{
        "in_reads": 2580480,
        "w_reads": 9289728,
        "out_reads": 11239424,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 12845056,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 495452160.0,
        "w_reads": 0,
        "out_reads": 627802112,
        "in_writes": 2580480,
        "w_writes": 0,
        "out_writes": 627802112,
        "last_out_reads": 11239424,
        "last_out_writes": 12845056
    },{
        "in_reads": 16647192576.0,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 1981808640.0,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 16647192576,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 74317824,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 18525782016,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 18492063744,
        "last_out_reads": 1844871168,
        "last_out_writes": 1883406336
    }]

correct_latency_eyeriss_conv_timeloop_3 = [{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 193536,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.13326719576719576,
        "ideal_bandwidth_update": 0.07407407407407407,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 9,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 6.4034391534391535,
        "ideal_bandwidth_update": 3.5555555555555554,
        "ideal_bandwidth_fill": 0.0796957671957672,
        "ideal_bandwidth_drain": 0.07407407407407407
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 9,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.11904761904761904,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.004464285714285715,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9997106481481481,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.11082175925925926,
        "ideal_bandwidth_drain": 0.11313657407407407
    }]

correct_mops_eyeriss_conv_additional_reuse = [{
        "in_reads": 29491200,
        "w_reads": 2080899072,
        "out_reads": 11239424,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 12845056,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 103219200.0,
        "w_reads": 0,
        "out_reads": 627802112,
        "in_writes": 29491200,
        "w_writes": 0,
        "out_writes": 627802112,
        "last_out_reads": 11239424,
        "last_out_writes": 12845056
    },{
        "in_reads": 16647192576.0,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 412876800.0,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 16647192576,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 16647192576,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 18525782016,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 18492063744,
        "last_out_reads": 1844871168,
        "last_out_writes": 1883406336
    }]

correct_latency_eyeriss_conv_additional_reuse = [{
        "latency_read_drain": 530407424.0,
        "latency_fill_update": 173408256,
        "cc_per_tile": 193536,
        "stall_cycles": 356999168.0,
        "ideal_bandwidth_read": 12.2348828420257,
        "ideal_bandwidth_update": 0.07407407407407407,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 9,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 4.141534391534392,
        "ideal_bandwidth_update": 3.5555555555555554,
        "ideal_bandwidth_fill": 0.23488284202569917,
        "ideal_bandwidth_drain": 0.07407407407407407
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 9,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.024801587301587304,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 1.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9997106481481481,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.11082175925925926,
        "ideal_bandwidth_drain": 0.11313657407407407
    }]

correct_mops_eyeriss_conv_input_bypass = [{
        "in_reads": 495452160.0,
        "w_reads": 9289728,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 1605632,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 9289728,
        "out_reads": 616562688,
        "in_writes": 0,
        "w_writes": 9289728,
        "out_writes": 616562688,
        "last_out_reads": 0,
        "last_out_writes": 1605632
    },{
        "in_reads": 16647192576.0,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 1981808640.0,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 16647192576,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 74317824,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 18492063744,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 18492063744,
        "last_out_reads": 1844871168,
        "last_out_writes": 1849688064
    }]

correct_latency_eyeriss_conv_input_bypass = [{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 193536,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 2.9107142857142856,
        "ideal_bandwidth_update": 0.009259259259259259,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 9,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 3.5998677248677247,
        "ideal_bandwidth_update": 3.5555555555555554,
        "ideal_bandwidth_fill": 0.05357142857142857,
        "ideal_bandwidth_drain": 0.009259259259259259
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 9,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.11904761904761904,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.004464285714285715,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9997106481481481,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.11082175925925926,
        "ideal_bandwidth_drain": 0.1111111111111111
    }]

correct_mops_simba_conv_timeloop_1 = [{
        "in_reads": 10506240,
        "w_reads": 297271296.0,
        "out_reads": 3211264,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 4816896,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 30965760.0,
        "w_reads": 0,
        "out_reads": 118816768,
        "in_writes": 10506240,
        "w_writes": 0,
        "out_writes": 118816768,
        "last_out_reads": 3211264,
        "last_out_writes": 4816896
    },{
        "in_reads": 8323596288.0,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 123863040.0,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 2378170368.0,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 297271296.0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 8792440832.0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 8779595776.0,
        "last_out_reads": 455999488,
        "last_out_writes": 475267072
    },{
        "in_reads": 0,
        "w_reads": 16647192576.0,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 2378170368.0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    }]

correct_latency_simba_conv_timeloop_1 = [{
        "latency_read_drain": 260112384,
        "latency_fill_update": 260112384,
        "cc_per_tile": 387072,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.1955939783320735,
        "ideal_bandwidth_update": 0.018518518518518517,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 260112384,
        "latency_fill_update": 260112384,
        "cc_per_tile": 193536,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.5573192239858906,
        "ideal_bandwidth_update": 0.4444444444444444,
        "ideal_bandwidth_fill": 0.052736835474930716,
        "ideal_bandwidth_drain": 0.018518518518518517
    },{
        "latency_read_drain": 260112384,
        "latency_fill_update": 260112384,
        "cc_per_tile": 126,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 2.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.02976190476190476,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 260112384,
        "latency_fill_update": 260112384,
        "cc_per_tile": 63,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.2857142857142857,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.03571428571428571,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 260112384,
        "latency_fill_update": 260112384,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9992283950617283,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.05478395061728395,
        "ideal_bandwidth_drain": 0.05709876543209877
    },{
        "latency_read_drain": 260112384,
        "latency_fill_update": 260112384,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.14285714285714285,
        "ideal_bandwidth_drain": 0.0
    }]

correct_mops_simba_conv_timeloop_2 = [{
        "in_reads": 14254080,
        "w_reads": 1179648.0,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 802816,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 14254080.0,
        "w_reads": 0,
        "out_reads": 102760448,
        "in_writes": 14254080,
        "w_writes": 0,
        "out_writes": 102760448,
        "last_out_reads": 0,
        "last_out_writes": 802816
    },{
        "in_reads": 924844032.0,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 114032640.0,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 264241152.0,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 1179648.0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 1951645696.0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 1951645696.0,
        "last_out_reads": 101957632,
        "last_out_writes": 102760448
    },{
        "in_reads": 0,
        "w_reads": 1849688064.0,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 264241152.0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    }]

correct_latency_simba_conv_timeloop_2 = [{
        "latency_read_drain": 115605504,
        "latency_fill_update": 115605504,
        "cc_per_tile": 112896,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.13350340136054423,
        "ideal_bandwidth_update": 0.006944444444444444,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 115605504,
        "latency_fill_update": 115605504,
        "cc_per_tile": 56448,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0052437641723355,
        "ideal_bandwidth_update": 0.8888888888888888,
        "ideal_bandwidth_fill": 0.12329931972789115,
        "ideal_bandwidth_drain": 0.006944444444444444
    },{
        "latency_read_drain": 115605504,
        "latency_fill_update": 115605504,
        "cc_per_tile": 1008,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.12329931972789115,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 115605504,
        "latency_fill_update": 115605504,
        "cc_per_tile": 126,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.14285714285714285,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.0006377551020408164,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 115605504,
        "latency_fill_update": 115605504,
        "cc_per_tile": 7,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9995659722222222,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.055121527777777776,
        "ideal_bandwidth_drain": 0.05555555555555555
    },{
        "latency_read_drain": 115605504,
        "latency_fill_update": 115605504,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.14285714285714285,
        "ideal_bandwidth_drain": 0.0
    }]

correct_mops_tpu_conv_timeloop_1 = [{
        "in_reads": 861184,
        "w_reads": 0,
        "out_reads": 0.0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 802816.0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 589824,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 7225344.0,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 861184,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 589824.0,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 589824,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 7225344.0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 7225344.0,
        "last_out_reads": 0.0,
        "last_out_writes": 802816.0
    },{
        "in_reads": 0,
        "w_reads": 1849688064.0,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 589824.0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    }]

correct_latency_tpu_conv_timeloop_1 = [{
        "latency_read_drain": 451584.0,
        "latency_fill_update": 451584.0,
        "cc_per_tile": 451584.0,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 1.90702947845805,
        "ideal_bandwidth_update": 1.7777777777777777,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 451584.0,
        "latency_fill_update": 451584.0,
        "cc_per_tile": 451584.0,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 1.3061224489795917,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 451584.0,
        "latency_fill_update": 147456.0,
        "cc_per_tile": 49152.0,
        "stall_cycles": 304128.0,
        "ideal_bandwidth_read": 49.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 5.840277777777778,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 147456.0,
        "latency_fill_update": 147456.0,
        "cc_per_tile": 9408,
        "stall_cycles": 119232.0,
        "ideal_bandwidth_read": 20.897959183673468,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 20.897959183673468,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 28224,
        "latency_fill_update": 28224,
        "cc_per_tile": 56,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.8888888888888888,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.1111111111111111
    },{
        "latency_read_drain": 28224,
        "latency_fill_update": 28224,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.00031887755102040814,
        "ideal_bandwidth_drain": 0.0
    }]

correct_mops_gemmini_conv_timeloop_1 = [{
        "in_reads": 23224320,
        "w_reads": 9289728,
        "out_reads": 0.0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 1605632.0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 1040449536.0,
        "w_reads": 9289728.0,
        "out_reads": 0,
        "in_writes": 23224320,
        "w_writes": 9289728,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 1040449536.0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 1040449536.0,
        "last_out_reads": 0.0,
        "last_out_writes": 1605632.0
    },{
        "in_reads": 0,
        "w_reads": 16647192576.0,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 9289728.0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    }]

correct_latency_gemmini_conv_timeloop_1 = [{
        "latency_read_drain": 65608704.0,
        "latency_fill_update": 65608704.0,
        "cc_per_tile": 32544.0,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 0.49557522123893805,
        "ideal_bandwidth_update": 0.024472850431552496,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 65608704.0,
        "latency_fill_update": 65028096,
        "cc_per_tile": 10752,
        "stall_cycles": 580608.0,
        "ideal_bandwidth_read": 16.142857142857142,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.5,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 65028096,
        "latency_fill_update": 65028096,
        "cc_per_tile": 896,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9984567901234568,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0015432098765432098
    },{
        "latency_read_drain": 65028096,
        "latency_fill_update": 65028096,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.0005580357142857143,
        "ideal_bandwidth_drain": 0.0
    }]

correct_mops_eyeriss_conv_timeloop_stride_1 = [{
        "in_reads": 4108288,
        "w_reads": 9437184,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 802816,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 81199104,
        "w_reads": 0,
        "out_reads": 205520896,
        "in_writes": 4108288,
        "w_writes": 0,
        "out_writes": 205520896,
        "last_out_reads": 0,
        "last_out_writes": 802816
    },{
        "in_reads": 1849688064,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 649592832,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 1849688064,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 66060288,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 2463842304,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 2463842304,
        "last_out_reads": 614154240,
        "last_out_writes": 616562688
    }]

correct_latency_eyeriss_conv_timeloop_stride_1 = [{
        "latency_read_drain": 17920000.0,
        "latency_fill_update": 17920000.0,
        "cc_per_tile": 70000.0,
        "stall_cycles": 0.0,
        "ideal_bandwidth_read": 0.7558857142857143,
        "ideal_bandwidth_update": 0.0448,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 17920000.0,
        "latency_fill_update": 13101824.0,
        "cc_per_tile": 6,
        "stall_cycles": 6909952.0,
        "ideal_bandwidth_read": 25.96875,
        "ideal_bandwidth_update": 18.666666666666668,
        "ideal_bandwidth_fill": 0.37313988095238093,
        "ideal_bandwidth_drain": 0.07291666666666667
    },{
        "latency_read_drain": 11010048,
        "latency_fill_update": 11010048,
        "cc_per_tile": 6,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.3511904761904762,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 11010048,
        "latency_fill_update": 11010048,
        "cc_per_tile": 2,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.03571428571428571,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 11010048,
        "latency_fill_update": 11010048,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9986979166666666,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.33203125,
        "ideal_bandwidth_drain": 0.3333333333333333
    }]

correct_mops_eyeriss_conv_timeloop_stride_2 = [{
        "in_reads": 9139200,
        "w_reads": 9289728,
        "out_reads": 24084480,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 25690112,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 495452160,
        "w_reads": 0,
        "out_reads": 332365824,
        "in_writes": 9139200,
        "w_writes": 0,
        "out_writes": 332365824,
        "last_out_reads": 24084480,
        "last_out_writes": 25690112
    },{
        "in_reads": 16647192576,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 1981808640,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 16647192576,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 74317824,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 17639473152,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 17567219712,
        "last_out_reads": 920027136,
        "last_out_writes": 997097472
    }]

correct_latency_eyeriss_conv_timeloop_stride_2 = [{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 96768,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.24516369047619047,
        "ideal_bandwidth_update": 0.14814814814814814,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 18,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 4.625661375661376,
        "ideal_bandwidth_update": 1.7777777777777777,
        "ideal_bandwidth_fill": 0.19159226190476192,
        "ideal_bandwidth_drain": 0.14814814814814814
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 18,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.11904761904761904,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.004464285714285715,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 173408256,
        "latency_fill_update": 173408256,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9997106481481481,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.055266203703703706,
        "ideal_bandwidth_drain": 0.059895833333333336
    }]

correct_mops_simba_conv_timeloop_stride_1 = [{
        "in_reads": 53616640,
        "w_reads": 74317824,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 1605632,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 346300416,
        "w_reads": 0,
        "out_reads": 693633024,
        "in_writes": 53616640,
        "w_writes": 0,
        "out_writes": 693633024,
        "last_out_reads": 0,
        "last_out_writes": 1605632
    },{
        "in_reads": 16647192576,
        "w_reads": 0,
        "out_reads": 0,
        "in_writes": 692600832,
        "w_writes": 0,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 1040449536,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 74317824,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    },{
        "in_reads": 0,
        "w_reads": 0,
        "out_reads": 5545852928,
        "in_writes": 0,
        "w_writes": 0,
        "out_writes": 5545852928,
        "last_out_reads": 1384054784,
        "last_out_writes": 1387266048
    },{
        "in_reads": 0,
        "w_reads": 16647192576,
        "out_reads": 0,
        "in_writes": 0,
        "w_writes": 1040449536,
        "out_writes": 0,
        "last_out_reads": 0,
        "last_out_writes": 0
    }]

correct_latency_simba_conv_timeloop_stride_1 = [{
        "latency_read_drain": 1040449536,
        "latency_fill_update": 1040449536,
        "cc_per_tile": 64512,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.12296075837742504,
        "ideal_bandwidth_update": 0.0015432098765432098,
        "ideal_bandwidth_fill": 0.0,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 1040449536,
        "latency_fill_update": 1040449536,
        "cc_per_tile": 21504,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.997960758377425,
        "ideal_bandwidth_update": 0.6666666666666666,
        "ideal_bandwidth_fill": 0.051532186948853614,
        "ideal_bandwidth_drain": 0.00154320987654321
    },{
        "latency_read_drain": 1040449536,
        "latency_fill_update": 1040449536,
        "cc_per_tile": 10752,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 4.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.16641865079365079,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 1040449536,
        "latency_fill_update": 1040449536,
        "cc_per_tile": 96,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.25,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.017857142857142856,
        "ideal_bandwidth_drain": 0.0
    },{
        "latency_read_drain": 1040449536,
        "latency_fill_update": 1040449536,
        "cc_per_tile": 16,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 0.9992283950617283,
        "ideal_bandwidth_update": 1.0,
        "ideal_bandwidth_fill": 0.33256172839506176,
        "ideal_bandwidth_drain": 0.3333333333333333
    },{
        "latency_read_drain": 1040449536,
        "latency_fill_update": 1040449536,
        "cc_per_tile": 1,
        "stall_cycles": 0,
        "ideal_bandwidth_read": 1.0,
        "ideal_bandwidth_update": 0.0,
        "ideal_bandwidth_fill": 0.0625,
        "ideal_bandwidth_drain": 0.0
    }]

# TESTS:

tests = [
    {
        "name": "gemmini_timeloop",
        "comp": Shape(
            M = 1024*3,
            K = 1024,
            N = 4096
        ),
        "bias_read": True,
        "arch": arch_gemmini_timeloop,
        "correct_mops": correct_mops_gemmini_timeloop,
        "correct_latency": correct_latency_gemmini_timeloop
    }, {
        "name": "gemmini_factorflow_1",
        "comp": Shape(
            M = 1024*3,
            K = 1024,
            N = 4096
        ),
        "bias_read": True,
        "arch": arch_gemmini_factorflow_1,
        "correct_mops": correct_mops_gemmini_factorflow_1,
        "correct_latency": correct_latency_gemmini_factorflow_1
    }, {
        "name": "gemmini_factorflow_2",
        "comp": Shape(
            M = 1024*3,
            K = 1024,
            N = 4096
        ),
        "bias_read": False,
        "arch": arch_gemmini_factorflow_2,
        "correct_mops": correct_mops_gemmini_factorflow_2,
        "correct_latency": correct_latency_gemmini_factorflow_2
    }, {
        "name": "eyeriss_timeloop",
        "comp": Shape(
            M = 1024*3,
            K = 1024,
            N = 4096
        ),
        "bias_read": False,
        "arch": arch_eyeriss_timeloop,
        "correct_mops": correct_mops_eyeriss_timeloop,
        "correct_latency": correct_latency_eyeriss_timeloop
    }, {
        "name": "eyeriss_factorflow_1",
        "comp": Shape(
            M = 1024*3,
            K = 1024,
            N = 4096
        ),
        "bias_read": False,
        "arch": arch_eyeriss_factorflow_1,
        "correct_mops": correct_mops_eyeriss_factorflow_1,
        "correct_latency": correct_latency_eyeriss_factorflow_1
    }, {
        "name": "eyeriss_timeloop_ex_1",
        "comp": Shape(
            M = 8,
            K = 1024,
            N = 8192
        ),
        "bias_read": False,
        "arch": arch_eyeriss_timeloop_ex_1,
        "correct_mops": correct_mops_eyeriss_timeloop_ex_1,
        "correct_latency": correct_latency_eyeriss_timeloop_ex_1
    }, {
        "name": "eyeriss_timeloop_ex_2",
        "comp": Shape(
            M = 1024*3,
            K = 1024,
            N = 4096
        ),
        "bias_read": False,
        "arch": arch_eyeriss_timeloop_ex_2,
        "correct_mops": correct_mops_eyeriss_timeloop_ex_2,
        "correct_latency": correct_latency_eyeriss_timeloop_ex_2
    }, {
        "name": "simba_timeloop",
        "comp": Shape(
            M = 1024*3,
            K = 1024,
            N = 4096
        ),
        "bias_read": False,
        "arch": arch_simba_timeloop,
        "correct_mops": correct_mops_simba_timeloop,
        "correct_latency": correct_latency_simba_timeloop
    }, {
        "name": "simba_factorflow_1",
        "comp": Shape(
            M = 1024*3,
            K = 1024,
            N = 4096
        ),
        "bias_read": False,
        "arch": arch_simba_factorflow_1,
        "correct_mops": correct_mops_simba_factorflow_1,
        "correct_latency": correct_latency_simba_factorflow_1
    }, {
        "name": "tpu_factorflow_1",
        "comp": Shape(
            M = 64,
            K = 4096,
            N = 4096
        ),
        "bias_read": False,
        "arch": arch_tpu_factorflow_1,
        "correct_mops": correct_mops_tpu_factorflow_1,
        "correct_latency": correct_latency_tpu_factorflow_1
    }, {
        "name": "tpu_timeloop_ex",
        "comp": Shape(
            M = 512,
            K = 256,
            N = 256
        ),
        "bias_read": False,
        "arch": arch_tpu_timeloop_ex,
        "correct_mops": correct_mops_tpu_timeloop_ex,
        "correct_latency": correct_latency_tpu_timeloop_ex
    }, {
        "name": "eyeriss_conv_timeloop_1",
        "comp": Shape(C = 256, M = 256, P = 56, Q = 56, R = 3, S = 3),
        "bias_read": False,
        "arch": arch_eyeriss_conv_timeloop_1,
        "correct_mops": correct_mops_eyeriss_conv_timeloop_1,
        "correct_latency": correct_latency_eyeriss_conv_timeloop_1
    }, {
        "name": "eyeriss_conv_timeloop_2",
        "comp": Shape(C = 256, M = 256, P = 56, Q = 56, R = 3, S = 3),
        "bias_read": False,
        "arch": arch_eyeriss_conv_timeloop_2,
        "correct_mops": correct_mops_eyeriss_conv_timeloop_2,
        "correct_latency": correct_latency_eyeriss_conv_timeloop_2
    }, {
        "name": "eyeriss_conv_timeloop_3",
        "comp": Shape(C = 128, M = 128, P = 112, Q = 112, R = 9, S = 9),
        "bias_read": False,
        "arch": arch_eyeriss_conv_timeloop_3,
        "correct_mops": correct_mops_eyeriss_conv_timeloop_3,
        "correct_latency": correct_latency_eyeriss_conv_timeloop_3
    }, {
        "name": "eyeriss_conv_additional_reuse",
        "comp": Shape(C = 128, M = 128, P = 112, Q = 112, R = 9, S = 9),
        "bias_read": False,
        "arch": arch_eyeriss_conv_additional_reuse,
        "correct_mops": correct_mops_eyeriss_conv_additional_reuse,
        "correct_latency": correct_latency_eyeriss_conv_additional_reuse
    }, {
        "name": "eyeriss_conv_input_bypass",
        "comp": Shape(C = 128, M = 128, P = 112, Q = 112, R = 9, S = 9),
        "bias_read": False,
        "arch": arch_eyeriss_conv_input_bypass,
        "correct_mops": correct_mops_eyeriss_conv_input_bypass,
        "correct_latency": correct_latency_eyeriss_conv_input_bypass
    }, {
        "name": "simba_conv_timeloop_1",
        "comp": Shape(C = 128, M = 128, P = 112, Q = 112, R = 9, S = 9),
        "bias_read": False,
        "arch": arch_simba_conv_timeloop_1,
        "correct_mops": correct_mops_simba_conv_timeloop_1,
        "correct_latency": correct_latency_simba_conv_timeloop_1
    }, {
        "name": "simba_conv_timeloop_2",
        "comp": Shape(C = 256, M = 256, P = 56, Q = 56, R = 3, S = 3),
        "bias_read": False,
        "arch": arch_simba_conv_timeloop_2,
        "correct_mops": correct_mops_simba_conv_timeloop_2,
        "correct_latency": correct_latency_simba_conv_timeloop_2
    }, {
        "name": "tpu_conv_timeloop_1",
        "comp": Shape(C = 256, M = 256, P = 56, Q = 56, R = 3, S = 3),
        "bias_read": False,
        "arch": arch_tpu_conv_timeloop_1,
        "correct_mops": correct_mops_tpu_conv_timeloop_1,
        "correct_latency": correct_latency_tpu_conv_timeloop_1
    }, {
        "name": "gemmini_conv_timeloop_1",
        "comp": Shape(C = 128, M = 128, P = 112, Q = 112, R = 9, S = 9),
        "bias_read": False,
        "arch": arch_gemmini_conv_timeloop_1,
        "correct_mops": correct_mops_gemmini_conv_timeloop_1,
        "correct_latency": correct_latency_gemmini_conv_timeloop_1
    }, {
        "name": "eyeriss_conv_timeloop_stride_1",
        "comp": Shape(C = 256, M = 256, P = 56, Q = 56, R = 3, S = 3, Pstride = 2, Qstride = 2, Rdilation = 3, Sdilation = 3),
        "bias_read": False,
        "arch": arch_eyeriss_conv_timeloop_stride_1,
        "correct_mops": correct_mops_eyeriss_conv_timeloop_stride_1,
        "correct_latency": correct_latency_eyeriss_conv_timeloop_stride_1
    }, {
        "name": "eyeriss_conv_timeloop_stride_2",
        "comp": Shape(C = 128, M = 128, P = 112, Q = 112, R = 9, S = 9, Pstride = 1, Qstride = 4, Rdilation = 1, Sdilation = 3),
        "bias_read": False,
        "arch": arch_eyeriss_conv_timeloop_stride_2,
        "correct_mops": correct_mops_eyeriss_conv_timeloop_stride_2,
        "correct_latency": correct_latency_eyeriss_conv_timeloop_stride_2,
        "change_settings": {"OVERESTIMATE_DISTINCT_VALUES": True}
    }, {
        "name": "simba_conv_timeloop_stride_1",
        "comp": Shape(C = 128, M = 128, P = 112, Q = 112, R = 9, S = 9, Pstride = 1, Qstride = 4, Rdilation = 1, Sdilation = 3),
        "bias_read": False,
        "arch": arch_simba_conv_timeloop_stride_1,
        "correct_mops": correct_mops_simba_conv_timeloop_stride_1,
        "correct_latency": correct_latency_simba_conv_timeloop_stride_1,
        "change_settings": {"OVERESTIMATE_DISTINCT_VALUES": True}
    }
]

if __name__ == "__main__":
    for test in tests:
        print(f"Test: {test['name']}")
        print("Computation: ", test['comp'])
        if "change_settings" in test:
            print("Modified Settings: ", test['change_settings'])
        print(f"Bias read: {test['bias_read']}")
        runTest(test['arch'], test['correct_mops'] + test['correct_latency'], test['comp'], test['bias_read'], test["change_settings"] if "change_settings" in test else None)
        print("PASSED!\n")