import numpy as np

from .common import read_json, write_json
from .trace_utils import words_to_time_segments


def visualize(ctx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import soundfile as sf
    from PIL import Image
    cfg = ctx.cfg['visualize']
    candidates = []
    for row in ctx.manifest():
        sid = row['sample_id']
        required = [('aligned', 'npz'), ('receptive', 'npz'), ('vision', 'npz'), ('audio', 'npz')]
        if all(ctx.sample_path(stage, sid, ext).exists() for stage, ext in required):
            audit = read_json(ctx.sample_path('aligned', sid))
            preferred = any(w['norm'].strip('.,!?;:') in cfg['preferred_terms'] or w['norm'].endswith("n't") for w in audit['words'])
            face_valid = float(np.load(ctx.sample_path('vision', sid, 'npz'))['valid'].mean())
            candidates.append((preferred, len(audit['words']) <= cfg['max_words'], face_valid, row))
    if cfg['sample_id']:
        candidates = [c for c in candidates if c[-1]['sample_id'] == cfg['sample_id']]
    if not candidates:
        raise ValueError('No complete real sample available for visualization')
    row = max(candidates, key=lambda x: x[:3])[-1]
    sid = row['sample_id']
    r = np.load(ctx.sample_path('receptive', sid, 'npz'))
    a = np.load(ctx.sample_path('aligned', sid, 'npz'))
    audit = read_json(ctx.sample_path('aligned', sid))
    words = audit['words']
    n = min(len(words), cfg['max_words'])
    names = [w['text'] for w in words[:n]]
    fig, ax = plt.subplots(figsize=(max(9, n * .28), max(7, n * .25)), layout='constrained')
    im = ax.imshow(r['R_word'][:n, :n], cmap='magma', aspect='auto', vmin=0)
    ax.set_xticks(range(n), names, rotation=90, fontsize=7)
    ax.set_yticks(range(n), names, fontsize=7)
    ax.set_xlabel('Masked input word'); ax.set_ylabel('Output word representation')
    ax.set_title('Word-level normalized occlusion sensitivity (not semantic percentage)')
    fig.colorbar(im, ax=ax, label='Sensitivity weight')
    fig.savefig(ctx.path('figures', 'word_receptive_field.png'), dpi=cfg['dpi']); plt.close(fig)
    marked = [i for i, w in enumerate(words[:-1]) if w['norm'].strip('.,!?;:') in cfg['preferred_terms'] or w['norm'].endswith("n't")]
    target = marked[0] + 1 if marked else next(i for i, w in enumerate(words) if w['time_valid'])
    positions = np.flatnonzero(a['token_word_idx'] == target)
    position = int(positions[0])
    projection = words_to_time_segments(r['R_occ'][position], words, normalize=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, layout='constrained')
    w = words[target]
    if w['time_valid']:
        axes[0].broken_barh([(w['start'], w['end'] - w['start'])], (0, 1), color='steelblue')
    axes[0].set_title(f'Own word anchor: {w["text"]}')
    for start, end, score, word in projection['segments']:
        axes[1].bar((start + end) / 2, score, width=end - start, alpha=.75)
    axes[1].set_title(f'Projected representation sensitivity; unmapped mass={projection["unmapped_mass"]:.4f}')
    axes[1].set_xlabel('Source clip presentation time (seconds)'); axes[1].set_ylabel('Word sensitivity mass')
    fig.savefig(ctx.path('figures', 'receptive_field_time_projection.png'), dpi=cfg['dpi']); plt.close(fig)
    pre = read_json(ctx.sample_path('preprocess', sid))
    wave, sr = sf.read(pre['audio_path'])
    audio = np.load(ctx.sample_path('audio', sid, 'npz'))
    vision = np.load(ctx.sample_path('vision', sid, 'npz'))
    fig = plt.figure(figsize=(14, 10), layout='constrained')
    grid = fig.add_gridspec(5, cfg['keyframes'], height_ratios=[1.2, 1, 1, 1, 1.6])
    axes = [fig.add_subplot(grid[i, :]) for i in range(4)]
    for i, word in enumerate(words):
        if word['time_valid']:
            axes[0].broken_barh([(word['start'], word['end'] - word['start'])], (i % 3, .7), facecolors='steelblue')
            axes[0].text(word['start'], i % 3 + .75, word['text'], fontsize=6, rotation=30)
    axes[0].set_ylim(0, 4); axes[0].set_title('Official words and forced-aligned time intervals')
    stride = max(1, len(wave) // 15000)
    axes[1].plot(np.arange(0, len(wave), stride) / sr, wave[::stride], linewidth=.4)
    axes[1].set_ylabel('Waveform')
    ai = read_json(ctx.sample_path('audio', sid))
    col = next((i for i, x in enumerate(ai['columns']) if 'Loudness' in x), 0)
    axes[2].plot(audio['times'], audio['features'][:, col], linewidth=.7)
    axes[2].set_ylabel(ai['columns'][col], fontsize=8)
    for mi, m in enumerate(['text', 'audio', 'vision']):
        for k, (start, end) in enumerate(a['token_time']):
            if start >= 0 and a[m + '_valid_mask'][k]:
                axes[3].broken_barh([(start, end - start)], (mi, .65))
    axes[3].set_yticks([.3, 1.3, 2.3], ['text anchors', 'audio', 'vision'])
    axes[3].set_xlabel('Source clip presentation time (seconds)')
    for ax in axes:
        ax.set_xlim(0, len(wave) / sr)
    good = np.flatnonzero(vision['valid'])
    selected = good[np.linspace(0, len(good) - 1, cfg['keyframes']).astype(int)] if len(good) else []
    import av
    images = {}
    selected_set = set(int(x) for x in selected)
    with av.open(row['video_path']) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in selected_set:
                images[index] = frame.to_image()
    for j, f in enumerate(selected):
        ax = fig.add_subplot(grid[4, j])
        ax.imshow(images[int(f)]); ax.axis('off'); ax.set_title(f'{vision["times"][f]:.3f} s')
    fig.savefig(ctx.path('figures', 'multimodal_alignment.png'), dpi=cfg['dpi']); plt.close(fig)
    write_json(ctx.path('meta', 'visualization_sample.json'), {'sample_id': sid, 'target_word_idx': target,
               'target_position': position, 'heatmap_cropped_to_words': n, 'keyframe_indices': selected})
