from factors import *
from architectures import *
from utils import *
from prints import prettyPrint
from main import updateStats

def runTest(arch, correct_data, comp, bias_read):
    initFactors(arch, comp)
    enforceFactorsConstraints(arch)
    assert checkDataflowConstraints(arch), "Dataflow constraints violated."
    setupBypasses(arch)
    updateInstances(arch)
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
    assert passed, "Test FAILED..."

def generateTestMOPs(arch):
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

def generateTestLatency(arch):
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

# TESTS:

tests = [
    {
        "name": "gemmini_timeloop",
        "comp": Shape(
            D = 1024*3,
            E = 1024,
            L = 4096
        ),
        "bias_read": True,
        "arch": arch_gemmini_timeloop,
        "correct_mops": correct_mops_gemmini_timeloop,
        "correct_latency": correct_latency_gemmini_timeloop
    }, {
        "name": "gemmini_factorflow_1",
        "comp": Shape(
            D = 1024*3,
            E = 1024,
            L = 4096
        ),
        "bias_read": True,
        "arch": arch_gemmini_factorflow_1,
        "correct_mops": correct_mops_gemmini_factorflow_1,
        "correct_latency": correct_latency_gemmini_factorflow_1
    }, {
        "name": "gemmini_factorflow_2",
        "comp": Shape(
            D = 1024*3,
            E = 1024,
            L = 4096
        ),
        "bias_read": False,
        "arch": arch_gemmini_factorflow_2,
        "correct_mops": correct_mops_gemmini_factorflow_2,
        "correct_latency": correct_latency_gemmini_factorflow_2
    }, {
        "name": "eyeriss_timeloop",
        "comp": Shape(
            D = 1024*3,
            E = 1024,
            L = 4096
        ),
        "bias_read": False,
        "arch": arch_eyeriss_timeloop,
        "correct_mops": correct_mops_eyeriss_timeloop,
        "correct_latency": correct_latency_eyeriss_timeloop
    }, {
        "name": "eyeriss_factorflow_1",
        "comp": Shape(
            D = 1024*3,
            E = 1024,
            L = 4096
        ),
        "bias_read": False,
        "arch": arch_eyeriss_factorflow_1,
        "correct_mops": correct_mops_eyeriss_factorflow_1,
        "correct_latency": correct_latency_eyeriss_factorflow_1
    }, {
        "name": "simba_timeloop",
        "comp": Shape(
            D = 1024*3,
            E = 1024,
            L = 4096
        ),
        "bias_read": False,
        "arch": arch_simba_timeloop,
        "correct_mops": correct_mops_simba_timeloop,
        "correct_latency": correct_latency_simba_timeloop
    }, {
        "name": "simba_factorflow_1",
        "comp": Shape(
            D = 1024*3,
            E = 1024,
            L = 4096
        ),
        "bias_read": False,
        "arch": arch_simba_factorflow_1,
        "correct_mops": correct_mops_simba_factorflow_1,
        "correct_latency": correct_latency_simba_factorflow_1
    }
]

if __name__ == "__main__":
    for test in tests:
        print(f"Test: {test['name']}")
        print("Computation: ", end='')
        prettyPrint(test['comp'])
        print(f"Bias read: {test['bias_read']}")
        runTest(test['arch'], test['correct_mops'] + test['correct_latency'], test['comp'], test['bias_read'])
        print("PASSED!\n")