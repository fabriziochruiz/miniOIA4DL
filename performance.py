
import math
import time
import numpy as np
import os
import csv
from datetime import datetime

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# NO TOCAR, NO HACE FALTA TOCAR NADA DE ESTE FICHERO
def compute_loss_and_gradient(predictions, labels):
    batch_size = len(predictions)
    loss = 0.0
    grad = []

    for pred, label in zip(predictions, labels):
        sample_loss = 0.0
        sample_grad = []
        for p, y in zip(pred, label):
            # Add small epsilon for numerical stability
            epsilon = 1e-9
            p = max(min(p, 1 - epsilon), epsilon)
            sample_loss += -y * math.log(p)
            sample_grad.append(p - y)
        loss += sample_loss
        grad.append(sample_grad)

    loss /= batch_size
    return loss, grad






def perf(model, train_images, train_labels, batch_size=64):
    num_samples = batch_size
    i=0
    batch_images = train_images[i:i+batch_size]

    start_time = time.time()
        
    output = batch_images
           
    output = model.forward(batch_images, curr_iter=i,training=False)
    
    duration = time.time() - start_time
    ips = num_samples / duration

    print(f"Total time: {duration:.2f}s IPS: {ips:.2f}images/sec")

    _save_profile_and_plot(
        model=model,
        batch_size=batch_size,
        total_time_s=duration,
        ips=ips,
    )


def _save_profile_and_plot(model, batch_size, total_time_s, ips):
    profile = getattr(model, 'last_fw_profile', None)
    if not profile:
        print("No se encontro perfil de capas para generar grafica.")
        return

    reports_dir = os.path.join('reports')
    os.makedirs(reports_dir, exist_ok=True)

    csv_path = os.path.join(reports_dir, 'performance_history.csv')
    run_label = _build_run_label(model)
    timestamp = datetime.now().isoformat(timespec='seconds')

    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'run_label',
                'timestamp',
                'layer_idx',
                'layer_name',
                'time_s',
                'batch_size',
                'total_time_s',
                'ips',
            ])
        for row in profile:
            writer.writerow([
                run_label,
                timestamp,
                row['layer_idx'],
                row['layer_name'],
                f"{row['time_s']:.10f}",
                batch_size,
                f"{total_time_s:.10f}",
                f"{ips:.10f}",
            ])

    if plt is None:
        print("Matplotlib no disponible. Se guardo historial en reports/performance_history.csv")
        print("Instala matplotlib en tu entorno actual y vuelve a ejecutar:sudo apt install python3-matplotlib")
        return

    _render_stacked_plot(csv_path, os.path.join(reports_dir, 'performance_stacked.png'))


def _build_run_label(model):
    custom_label = getattr(model, 'run_label', None)
    if custom_label:
        return custom_label

    conv_algo = getattr(model, 'conv_algo', None)
    if conv_algo is None:
        return datetime.now().strftime('RUN_%m%d_%H%M%S')

    algo_labels = {
        0: 'BASE',
        1: 'I2C',
        2: 'ALG2',
    }
    base = algo_labels.get(conv_algo, f'ALG{conv_algo}')
    return f"{base}_{datetime.now().strftime('%H%M%S')}"


def _render_stacked_plot(csv_path, image_path):
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    run_order = []
    run_seen = set()
    component_order = []
    component_seen = set()
    data = {}

    for row in rows:
        run_label = row['run_label']
        layer_idx = int(row['layer_idx'])
        layer_name = row['layer_name']
        component_key = f"{layer_idx:03d}:{layer_name}"
        time_s = float(row['time_s'])

        if run_label not in run_seen:
            run_order.append(run_label)
            run_seen.add(run_label)
            data[run_label] = {}

        if component_key not in component_seen:
            component_order.append(component_key)
            component_seen.add(component_key)

        data.setdefault(run_label, {})
        data[run_label][component_key] = data[run_label].get(component_key, 0.0) + time_s

    totals = {}
    for run_label in run_order:
        totals[run_label] = sum(data[run_label].values())

    x = np.arange(len(run_order))
    bottom = np.zeros(len(run_order), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 6))
    layer_colors = _build_distinct_palette(len(component_order))

    for color_idx, component_key in enumerate(component_order):
        values_pct = []
        for run_label in run_order:
            total = totals[run_label]
            value = data[run_label].get(component_key, 0.0)
            pct = 100.0 * value / total if total > 0 else 0.0
            values_pct.append(pct)
        layer_idx_str, layer_name = component_key.split(':', 1)
        label_name = f"L{int(layer_idx_str)} {layer_name}"
        ax.bar(
            x,
            values_pct,
            bottom=bottom,
            width=0.7,
            label=label_name,
            color=layer_colors[color_idx],
            edgecolor='white',
            linewidth=0.3,
        )
        bottom += np.array(values_pct)

    ax.set_ylim(0, 100)
    ax.set_ylabel('Contribution (%)')
    ax.set_title('Layer Time Distribution per Run')
    ax.set_xticks(x)
    ax.set_xticklabels(run_order, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    print(f"Grafica actualizada en {image_path}")


def _build_distinct_palette(count):
    if count <= 0:
        return []

    colors = []
    for cmap_name in ('tab20', 'tab20b', 'tab20c'):
        cmap = plt.get_cmap(cmap_name)
        for i in range(20):
            colors.append(cmap(i / 19))

    if count <= len(colors):
        return colors[:count]

    extra = count - len(colors)
    hsv = plt.get_cmap('hsv')
    for i in range(extra):
        colors.append(hsv(i / max(extra, 1)))

    return colors[:count]

    