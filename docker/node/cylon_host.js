// Mock cylon_host module for local-only WASM operations
// This provides stub implementations for host imports when FMI is not available
// For actual distributed operations, use CylonWasmHost which provides real implementations

module.exports = {
  host_get_rank: () => 0,
  host_get_world_size: () => 1,
  host_barrier: () => {},
  host_all_to_all: (partitions_ptr, partition_lens_ptr, num_partitions, results_ptr_out, num_results_out) => {
    throw new Error('FMI not available in local mode - distributed operations require multi-process setup');
  },
  host_all_gather: (data_ptr, data_len, results_ptr_out, num_results_out) => {
    throw new Error('FMI not available in local mode - distributed operations require multi-process setup');
  },
  host_broadcast: (root, data_ptr, data_len, result_ptr_out, result_len_out) => {
    throw new Error('FMI not available in local mode - distributed operations require multi-process setup');
  },
  host_gather: (root, data_ptr, data_len, results_ptr_out, num_results_out) => {
    throw new Error('FMI not available in local mode - distributed operations require multi-process setup');
  },
  host_scatter: (root, partitions_ptr, partition_lens_ptr, num_partitions, result_ptr_out, result_len_out) => {
    throw new Error('FMI not available in local mode - distributed operations require multi-process setup');
  },
};