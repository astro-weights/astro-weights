A tool call the server cannot parse (llama-server answers HTTP 500 "Failed to parse tool call") is now a BFCL turn without a call, not a retried error that aborts `run_bfcl` and loses the whole run.
