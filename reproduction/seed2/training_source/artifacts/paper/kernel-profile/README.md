# CUDA inference profile

`profile-optimized.json` is the submission profile produced from a clean
checkout on one NVIDIA L4.  The timed scope is exact forest inference; input,
candidate-factor, and forest construction are excluded.  Dense and low-rank
backends are warmed up and measured separately.  Before each backend's peak is
reset, only that backend's required steady-state inputs remain live.

The JSON records raw timings, allocator peaks, logical input bytes, software
versions, agreement with the reference implementation, and the complete
measurement protocol.  At length 1024 and candidate width 128, the measured
low-rank backend is 10.31 times faster and uses 4.52 times less peak allocator
memory than the pre-materialized dense backend under this protocol.
