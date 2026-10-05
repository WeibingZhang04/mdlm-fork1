"""Trace attribution must follow launches, including asynchronous Triton work."""
import pytest

from chain_crf.profiling import summarize_trace, format_stage_report


def event(cat, name, ts, dur=1, tid=1, pid=100, correlation=None, external=None):
    args = {}
    if correlation is not None:
        args['correlation'] = correlation
    if external is not None:
        args['External id'] = external
    return dict(ph='X', cat=cat, name=name, ts=ts, dur=dur, tid=tid, pid=pid, args=args)


def test_runtime_and_driver_launches_include_ancestors_without_external_id_collisions():
    events = [
        event('user_annotation', 'crf.inference', 0, 100),
        event('user_annotation', 'crf.forward_filter', 10, 20),
        event('user_annotation', 'crf.backward_sample', 40, 30),
        event('cuda_driver', 'cuLaunchKernel', 15, correlation=51, external=99),
        event('cuda_runtime', 'cudaLaunchKernel', 45, correlation=52, external=100),
        event('cuda_runtime', 'cudaMemcpyAsync', 20, correlation=53),
        event('cuda_runtime', 'cudaMemsetAsync', 21, correlation=54),
        # A colliding external ID must not put the Triton kernel in backward.
        event('cpu_op', 'aten::unsqueeze', 46, external=99),
        # Execution is delayed past both CPU ranges; GPU timestamp containment
        # would attribute these incorrectly or miss them entirely.
        event('kernel', '_forward_kernel', 200, 8, pid=0, tid=7, correlation=51, external=99),
        event('kernel', 'pytorch_kernel', 220, 4, pid=0, tid=7, correlation=52, external=100),
        event('gpu_memcpy', 'copy', 230, 2, pid=0, tid=7, correlation=53),
        event('gpu_memset', 'memset', 240, 3, pid=0, tid=7, correlation=54),
        # GPU range annotations must not duplicate CPU stage rows/durations.
        event('gpu_user_annotation', 'crf.forward_filter', 190, 100, pid=0, tid=7),
    ]
    summary = summarize_trace({'traceEvents': list(reversed(events))})
    stages = {s['stage']: s for s in summary['stages']}
    assert len(stages) == 3
    forward, backward, parent = (stages[n] for n in
        ('crf.forward_filter', 'crf.backward_sample', 'crf.inference'))
    assert forward['calls'] == 1 and forward['cpu_total_ms'] == .02
    assert forward['device_kernel_ms'] == .008
    assert forward['device_memory_ms'] == .005
    assert forward['device_total_ms'] == pytest.approx(.013)
    assert backward['device_total_ms'] == .004
    assert parent['device_total_ms'] == pytest.approx(.017)
    assert parent['kernel_calls'] == 2
    assert summary['attribution']['attributed_events'] == 4
    assert 'GPU kernels' in format_stage_report(summary)


def test_thread_containment_unassigned_events_and_ambiguous_correlations():
    events = [
        event('user_annotation', 'crf.forward_filter', 0, 10, tid=1),
        event('user_annotation', 'crf.backward_sample', 0, 10, tid=2),
        event('cuda_driver', 'cuLaunchKernel', 2, tid=2, correlation=1),
        # Same kernel name does not imply forward-stage ownership.
        event('kernel', '_forward_kernel', 50, 2, pid=0, correlation=1),
        event('kernel', 'missing', 60, 3, pid=0, correlation=2),
        event('cuda_runtime', 'cudaLaunchKernel', 20, correlation=3),
        event('kernel', 'outside', 70, 4, pid=0, correlation=3),
        event('cuda_driver', 'cuLaunchKernel', 3, tid=1, correlation=4),
        event('cuda_driver', 'cuLaunchKernel', 3, tid=2, correlation=4),
        event('kernel', 'ambiguous', 80, 5, pid=0, correlation=4),
    ]
    summary = summarize_trace({'traceEvents': events})
    stages = {s['stage']: s for s in summary['stages']}
    assert stages['crf.forward_filter']['device_total_ms'] == 0
    assert stages['crf.backward_sample']['device_total_ms'] == .002
    attribution = summary['attribution']
    assert attribution['device_events'] == 4 and attribution['attributed_events'] == 1
    for name, duration in [('missing_launch', .003), ('outside_named_stages', .004),
                           ('ambiguous_launch', .005)]:
        assert attribution[name] == dict(events=1, device_total_ms=duration)


def test_repeated_and_nested_same_name_ranges_do_not_double_count_gpu_work():
    events = [
        event('user_annotation', 'crf.forward_filter', 0, 10),
        event('user_annotation', 'crf.forward_filter', 2, 3),
        event('user_annotation', 'crf.forward_filter', 20, 10),
        event('cuda_driver', 'cuLaunchKernel', 3, correlation=1),
        event('cuda_driver', 'cuLaunchKernel', 22, correlation=2),
        event('kernel', '_forward_kernel', 40, 4, pid=0, correlation=1),
        event('kernel', '_forward_kernel', 50, 6, pid=0, correlation=2),
    ]
    stage, = summarize_trace({'traceEvents': events})['stages']
    assert stage['calls'] == 3 and stage['cpu_total_ms'] == .023
    assert stage['kernel_calls'] == 2 and stage['device_kernel_ms'] == .010


def test_cpu_only_trace_reports_zero_device_work():
    summary = summarize_trace({'traceEvents': [event('user_annotation', 'mdlm.forward', 1, 25)]})
    assert summary['stages'][0]['device_total_ms'] == 0
    assert summary['attribution']['device_events'] == 0
