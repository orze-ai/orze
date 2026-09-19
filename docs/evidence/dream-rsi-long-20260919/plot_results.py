"""Standalone research figure from the published summary; no scientific reruns."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
s = json.loads((folder / 'summary.json').read_text())
fig, ax = plt.subplots(figsize=(8.8, 4.7), layout='constrained')
labels = ['Extrapolation', 'Interaction', 'Dynamics', 'Robustness', 'Transport']
for y, group in enumerate(s['problem_means']):
    delta = group['delta'] * 100
    ax.barh(y, delta, height=.4, color='#267eab' if delta > 0 else '#b95e51', alpha=.8)
    values = [r['delta'] * 100 for r in s['paired'] if r['problem'] == group['problem']]
    ax.scatter(values, [y, y], color='#28333b', s=26, zorder=3,
               label='Individual paired repetitions' if y == 0 else None)
lo, hi = [x * 100 for x in s['problem_t_95_interval']]
mean = s['mean_delta'] * 100
ax.errorbar(mean, 5.3, xerr=[[mean-lo], [hi-mean]], fmt='D', color='#222222',
            capsize=5, label='Equal-problem mean and 95% t interval')
ax.axvline(0, color='#555555', linewidth=.8)
ax.set_yticks([0, 1, 2, 3, 4, 5.3], labels + ['Overall mean'])
ax.invert_yaxis()
ax.set_xlabel('Dream minus control: confirmation RMSE reduction (percentage points)')
ax.set_title('No mean improvement established in the prospective comparison', loc='left', pad=16)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.grid(axis='x', alpha=.17)
ax.set_axisbelow(True)
ax.legend(loc='lower right', fontsize=8)
fig.get_layout_engine().set(rect=(0, .08, 1, 1))
fig.text(.02, .018, '5 synthetic problem groups × 2 paired data repetitions; 23.29 hours. '
         'Positive favors Dream.\nMean difference −0.32 pp; 95% interval [−5.57, +4.94] pp. '
         'Two repetitions are not two independent problem groups.', fontsize=8)
for ext in ('png', 'pdf'):
    fig.savefig(folder / f'mean-comparison.{ext}', dpi=180, bbox_inches='tight')
