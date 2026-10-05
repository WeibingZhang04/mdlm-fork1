"""Attribute Chrome-trace GPU activity by launch correlation, not External id.

PyTorch's native event tree can miss Triton driver launches or associate them
with an unrelated CPU op whose External id happens to collide. A GPU event's
correlation links it to its CUDA runtime/driver call. The CPU thread and launch
timestamp then identify the enclosing named stages, even for asynchronous work
that completes after those stages have ended.
"""
from bisect import bisect_right
from collections import defaultdict


PREFIXES = ('mdlm.', 'crf.', 'generation.')
NOTE = ('CPU totals include dispatch/waits. Device totals sum CUDA kernel, copy, '
        'and memset durations attributed by launch correlation and CPU-stage '
        'containment; they are not elapsed GPU wall time. Nested stages overlap. '
        'Do not add CPU and GPU times. Profiling adds overhead.')


def summarize_trace(trace):
    """Return inclusive named-stage totals and explicit attribution diagnostics.

    GPU annotations are excluded. Each GPU activity is counted once per stage
    name, including enclosing stages. Ambiguous/missing correlations remain
    unassigned instead of being guessed from names, GPU timestamps or IDs.
    """
    events = trace['traceEvents']
    stages = {}
    regions = defaultdict(lambda: defaultdict(list))
    for event in events:
        name = event.get('name', '')
        if (event.get('ph') != 'X' or event.get('cat') != 'user_annotation'
                or not name.startswith(PREFIXES)):
            continue
        stage = stages.setdefault(name, dict(stage=name, calls=0, cpu_total_ms=0.,
                                            device_total_ms=0., device_kernel_ms=0.,
                                            device_memory_ms=0., kernel_calls=0))
        stage['calls'] += 1
        stage['cpu_total_ms'] += event['dur'] / 1000
        regions[(event.get('pid'), event.get('tid'))][name].append(
            (event['ts'], event['ts'] + event['dur']))

    # Merge same-name overlapping regions to avoid double-counting GPU events
    # in recursive annotations and permit binary-search interval lookup.
    indexed = {}
    for thread, named in regions.items():
        indexed[thread] = {}
        for name, intervals in named.items():
            merged = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            indexed[thread][name] = ([start for start, _ in merged], merged)

    launches = defaultdict(set)
    for event in events:
        if event.get('ph') != 'X' or event.get('cat') not in ('cuda_runtime', 'cuda_driver'):
            continue
        correlation = event.get('args', {}).get('correlation')
        if correlation is None:
            continue
        names = []
        for name, (starts, intervals) in indexed.get((event.get('pid'), event.get('tid')), {}).items():
            index = bisect_right(starts, event['ts']) - 1
            if index >= 0 and event['ts'] < intervals[index][1]:
                names.append(name)
        launches[correlation].add(tuple(sorted(names)))

    diagnostics = {key: dict(events=0, device_total_ms=0.) for key in
                   ('missing_launch', 'ambiguous_launch', 'outside_named_stages')}
    device_events = attributed_events = 0
    for event in events:
        category = event.get('cat')
        if event.get('ph') != 'X' or category not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        device_events += 1
        duration = event['dur'] / 1000
        matches = launches.get(event.get('args', {}).get('correlation'), set())
        reason = ('missing_launch' if not matches else
                  'ambiguous_launch' if len(matches) != 1 else
                  'outside_named_stages' if not next(iter(matches)) else None)
        if reason:
            diagnostics[reason]['events'] += 1
            diagnostics[reason]['device_total_ms'] += duration
            continue
        attributed_events += 1
        for name in next(iter(matches)):
            stage = stages[name]
            stage['device_total_ms'] += duration
            stage['device_kernel_ms' if category == 'kernel' else 'device_memory_ms'] += duration
            if category == 'kernel':
                stage['kernel_calls'] += 1
    return dict(stages=list(stages.values()), note=NOTE,
                attribution=dict(method='cuda_launch_correlation_v1',
                                 device_events=device_events, attributed_events=attributed_events,
                                 **diagnostics))


def format_stage_report(summary):
    """Readable totals shared by new profiles and repaired saved traces."""
    lines = ['Profiling adds overhead; these are not clean benchmark timings.', NOTE,
             '', f'{"Stage":<28} {"Calls":>6} {"CPU ms":>11} {"GPU total":>11} '
             f'{"GPU kernels":>12} {"GPU memory":>11}']
    for stage in summary['stages']:
        lines.append(f"{stage['stage']:<28} {stage['calls']:>6} {stage['cpu_total_ms']:>11.3f} "
                     f"{stage['device_total_ms']:>11.3f} {stage['device_kernel_ms']:>12.3f} "
                     f"{stage['device_memory_ms']:>11.3f}")
    lines += ['', 'GPU columns are milliseconds; total = kernels + copies/memsets.',
              'Nested stages include their children; do not sum the rows.']
    for key in ('missing_launch', 'ambiguous_launch', 'outside_named_stages'):
        item = summary['attribution'][key]
        lines.append(f"Unassigned ({key}): {item['events']} events, {item['device_total_ms']:.3f} ms")
    return '\n'.join(lines)
