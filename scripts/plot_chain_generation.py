#!/usr/bin/env python3
"""Plot measured generation quality against network calls and elapsed time.

The specification contains a list of {label, runs}; each run is an evaluator
output directory. Only complete unconditional, equally sized evaluations enter
one plot. Published plot data contain numeric measurements and source hashes,
not machine-specific paths. No interpolation, seed aggregation or error bars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_point(directory):
    directory = Path(directory)
    manifest_path = directory/'manifest.json'
    metrics_path = directory/'metrics.json'
    scores_path = directory/'gpt2-large.json'
    records_path = directory/'samples.jsonl'
    manifest = json.loads(manifest_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    scores = json.loads(scores_path.read_text())
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    config = manifest.get('config', manifest.get('arguments'))
    if not config or config.get('synthetic') or manifest.get('backbone', {}).get('synthetic_only'):
        raise ValueError('A real model manifest is required')
    count, length, steps = config['samples'], config['length'], config['steps']
    if count != len(records) or metrics['samples'] != count or len(scores['samples']) != count:
        raise ValueError('Incomplete or mismatched sample counts')
    if [row['sample_id'] for row in records] != list(range(count)):
        raise ValueError('Samples must have unique contiguous local IDs')
    if any(row.get('prefix_length', 0) != 0 or len(row['token_ids']) != length for row in records):
        raise ValueError('This plot requires fixed-length unconditional generations')
    if scores['scored_tokens'] != count*(length-1):
        raise ValueError('Expected the common scorer over all raw tokens after the first')
    if any(row['scored_tokens'] != length-1 for row in scores['samples']):
        raise ValueError('Individual scorer records do not match fixed output length')
    values = [metrics['seconds_per_sample'], scores['perplexity'], metrics['token_entropy_nats'],
              metrics['within_sample_repeat_4']]
    if not all(math.isfinite(value) for value in values) or min(values[:2]) <= 0:
        raise ValueError('Nonfinite or invalid quality/time measurements')
    return {'steps': steps, 'samples': count, 'length': length, 'batch_size': config['batch_size'],
            'seconds_per_sample': values[0], 'generative_perplexity': values[1],
            'token_entropy_nats': values[2], 'repeat4': values[3],
            'evaluator': scores['model'], 'evaluator_revision': scores['requested_revision'],
            'manifest_sha256': digest(manifest_path), 'metrics_sha256': digest(metrics_path),
            'scores_sha256': digest(scores_path), 'samples_sha256': digest(records_path)}


def collect(specification):
    output = []
    labels = set()
    protocol = None
    for series in specification:
        label = series['label']
        if not isinstance(label, str) or not label or label in labels or not series['runs']:
            raise ValueError('Each series needs a unique nonempty label and runs')
        labels.add(label)
        points = sorted((load_point(path) for path in series['runs']), key=lambda point: point['steps'])
        if len({point['steps'] for point in points}) != len(points):
            raise ValueError('Each series may contain only one run per decoding budget')
        for point in points:
            current = tuple(point[key] for key in ('samples', 'length', 'batch_size',
                                                    'evaluator', 'evaluator_revision'))
            if protocol is not None and current != protocol:
                raise ValueError('Mixing sample counts, lengths, batch sizes or scorers in one plot')
            protocol = current
        output.append({'label': label, 'points': points})
    if not output:
        raise ValueError('Need at least one completed series')
    return output


def render(series, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new plot directory to preserve existing evidence')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = ['#222222', '#007A78', '#B06B00', '#6655A8', '#2866B3', '#BD4A50']
    markers = ['o', 's', '^', 'D', 'v', 'P']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), layout='constrained')
    for index, row in enumerate(series):
        points = row['points']
        style = {'label': row['label'], 'color': colors[index % len(colors)],
                 'marker': markers[index % len(markers)], 'markersize': 4,
                 'linewidth': 1.3}
        axes[0].plot([p['steps'] for p in points], [p['generative_perplexity'] for p in points], **style)
        # Connect observations in budget order, not a fitted or Pareto envelope.
        axes[1].plot([p['seconds_per_sample'] for p in points],
                     [p['generative_perplexity'] for p in points], **style)
    for axis in axes:
        axis.set_xscale('log', base=2)
        axis.set_yscale('log')
        axis.yaxis.set_major_formatter(ScalarFormatter())
        axis.grid(True, which='major', alpha=.18, linewidth=.6)
        axis.spines[['top', 'right']].set_visible(False)
    steps = sorted({p['steps'] for row in series for p in row['points']})
    axes[0].set_xticks(steps, labels=[str(step) for step in steps])
    axes[0].set_xlabel('Denoising steps')
    axes[1].set_xlabel('Seconds per sample (complete generation)')
    axes[0].set_ylabel('Generative perplexity (lower is better)')
    axes[1].legend(frameon=False)
    fig.suptitle(title, fontsize=10)
    for extension in ('pdf', 'png'):
        fig.savefig(output/f'quality-time.{extension}', dpi=220)
    plt.close(fig)
    (output/'plot-data.json').write_text(json.dumps({'title': title, 'series': series}, indent=2)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', required=True)
    args = parser.parse_args(argv)
    render(collect(json.loads(args.specification.read_text())), args.output, args.title)


if __name__ == '__main__':
    main()
