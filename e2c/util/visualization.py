import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
sns.set(style="whitegrid")


def smooth_curve(seq, chunk_size=5):
    n = len(seq)
    smoothed = []
    for i in range(0, n):
        smoothed.append(np.mean(seq[i:i + chunk_size]))
    return np.array(smoothed)


def load_results(result_path):
    if not os.path.isfile(result_path):
        dir_path = result_path
        result_files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.startswith("result") and f.endswith(".json")
        ]
        results = []
        for file in result_files:
            with open(file, "r") as f:
                results.extend(json.load(f))
    else:
        with open(result_path, "r") as f:
            results = json.load(f)
    print(f"Loaded {len(results)} results from {result_path}")
    return results


def plot_entropy_confidence(all_entropies, all_confidences, output_dir):
    max_len = max(len(seq) for seq in all_entropies)

    entropy_matrix = np.zeros((len(all_entropies), max_len))
    mask_matrix = np.zeros((len(all_entropies), max_len))
    for i, seq in enumerate(all_entropies):
        seq = np.abs(seq)
        entropy_matrix[i, :len(seq)] = seq
        mask_matrix[i, :len(seq)] = 1
    avg_entropy = np.sum(entropy_matrix, axis=0) / np.sum(mask_matrix, axis=0)

    plt.figure(figsize=(12, 5))
    idx = np.random.choice(len(all_entropies))
    seq_entropy = np.abs(all_entropies[idx])
    plt.plot(seq_entropy, alpha=0.5, linewidth=1.5, label="Sample (raw)")
    plt.plot(smooth_curve(seq_entropy), alpha=0.7, linewidth=2, label="Sample (smoothed)")
    plt.plot(smooth_curve(avg_entropy), color='blue', linewidth=2.5, label="Average (smoothed)")
    plt.xlabel("Token Index")
    plt.ylabel("Entropy (abs)")
    plt.title("Entropy Across Token Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "entropy_over_idx.png"))
    plt.close()

    confidence_matrix = np.zeros((len(all_confidences), max_len))
    for i, seq in enumerate(all_confidences):
        seq = np.abs(seq)
        confidence_matrix[i, :len(seq)] = seq
    avg_confidence = np.sum(confidence_matrix, axis=0) / np.sum(mask_matrix, axis=0)

    plt.figure(figsize=(12, 5))
    idx = np.random.choice(len(all_confidences))
    seq_conf = np.abs(all_confidences[idx])
    plt.plot(seq_conf, alpha=0.5, linewidth=1.5, label="Sample (raw)")
    plt.plot(smooth_curve(seq_conf), alpha=0.7, linewidth=2, label="Sample (smoothed)")
    plt.plot(smooth_curve(avg_confidence), color='orange', linewidth=2.5, label="Average (smoothed)")
    plt.xlabel("Token Index")
    plt.ylabel("Confidence (abs)")
    plt.title("Confidence Across Token Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_over_idx.png"))
    plt.close()


def plot_success_distributions(step_nums, lengths, avg_entropies, avg_confidences, success_flags, output_dir):
    def hist_plot(data, success, fail, xlabel, title, filename):
        plt.figure(figsize=(10, 5))
        sns.histplot(success, color="green", label="Success", kde=False, stat="density", bins=20)
        sns.histplot(fail, color="red", label="Fail", kde=False, stat="density", bins=20)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    hist_plot(step_nums, step_nums[success_flags], step_nums[~success_flags],
              "Max Step Number", "Step Number Distribution by Success", "stepnum_success.png")
    hist_plot(lengths, lengths[success_flags], lengths[~success_flags],
              "Sequence Length", "Sequence Length Distribution by Success", "length_success.png")
    hist_plot(avg_entropies, avg_entropies[success_flags], avg_entropies[~success_flags],
              "Average Entropy per Sequence", "Average Entropy Distribution by Success", "avg_entropy_success.png")
    hist_plot(avg_confidences, avg_confidences[success_flags], avg_confidences[~success_flags],
              "Average Confidence per Sequence", "Average Confidence Distribution by Success", "avg_confidence_success.png")


def plot_by_task_samples(results, output_dir, num_tasks=2, smooth_chunk=30):
    """Plot entropy and confidence trajectories for each task, colored by sample index."""
    task_data = []
    for item in results:
        task_samples = []
        for sample in item["samples"]:
            static = sample["static"]
            if static is None:
                continue
            if "confidences_all" in static and "entropies" in static:
                task_samples.append({
                    "confidences": np.array(static["confidences_all"]),
                    "entropies": np.array(static["entropies"]),
                    "length": static['length'],
                    "success": sample.get("success", False),
                })
        if task_samples:
            task_data.append({
                "samples": task_samples,
                "median_length": item.get('avg_length', 0),
            })

    selected_tasks = task_data[:num_tasks]
    cm_blues = plt.get_cmap('Blues')

    for tid, task in enumerate(selected_tasks, 1):
        plt.figure(figsize=(12, 6))
        num_samples = len(task["samples"])

        for i, sample in enumerate(task["samples"]):
            color_ent = cm_blues(0.2 + 0.8 * (i / (num_samples - 1) if num_samples > 1 else 0.5))
            marker = 'o' if sample["success"] else 'x'
            markersize = 8 if sample["success"] else 6

            if len(sample["confidences"]) > smooth_chunk:
                sample["confidences"] = smooth_curve(sample["confidences"], chunk_size=smooth_chunk)
            if len(sample["entropies"]) > smooth_chunk:
                sample["entropies"] = smooth_curve(sample["entropies"], chunk_size=smooth_chunk)

            plt.plot(sample["entropies"], color=color_ent, linestyle='--', alpha=0.7, label=f"Sample {i+1} - Entropy")
            plt.plot(len(sample["entropies"]) - 1, sample["entropies"][-1], marker=marker, markersize=markersize, color=color_ent)
            plt.axvline(x=sample["length"], color=color_ent, linestyle=':', alpha=0.5)

        legend_lines_ent = [plt.Line2D([0], [0], color=cm_blues(0.5), lw=2, linestyle='--')]
        legend_lines_success = [plt.Line2D([0], [0], color='k', marker='o', markersize=8, linestyle='None')]
        legend_lines_fail = [plt.Line2D([0], [0], color='k', marker='x', markersize=6, linestyle='None')]

        legend1 = plt.legend(legend_lines_ent, ["Entropy Trajectories"], loc='upper left')
        plt.gca().add_artist(legend1)
        plt.legend(legend_lines_success + legend_lines_fail, ["Success", "Failure"], loc='upper right')

        plt.xlabel("Token Index")
        plt.ylabel("Value")
        plt.title(f"Entropy Trajectories for Task {tid}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"task_{tid}_all_samples_trajectories.png"))
        plt.close()


def visualize_result_json(cfg):
    results = load_results(cfg.result_path)
    output_dir = cfg.output_dir
    num_tasks = cfg.num_tasks

    all_entropies, all_confidences, step_nums, lengths, avg_confidences, success_flags = [], [], [], [], [], []
    for item in results:
        for sample in item["samples"]:
            static = sample["static"]
            if static is None:
                continue
            all_entropies.append(static["entropies"])
            all_confidences.append(static["confidences_all"])
            lengths.append(static["length"])
            avg_confidences.append(static["confidences"]["seq_avg_confidence"])
            step_nums.append(min(10, static["max_step"]))
            success_flags.append(sample["success"])

    step_nums = np.array(step_nums)
    lengths = np.array(lengths)
    avg_confidences = np.array(avg_confidences)
    avg_entropies = np.array([np.mean(seq) for seq in all_entropies])
    success_flags = np.array(success_flags)

    os.makedirs(output_dir, exist_ok=True)
    plot_entropy_confidence(all_entropies, all_confidences, output_dir)
    plot_success_distributions(step_nums, lengths, avg_entropies, avg_confidences, success_flags, output_dir)
    plot_by_task_samples(results, output_dir, num_tasks=num_tasks)
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_tasks", type=int, default=5)
    visualize_result_json(parser.parse_args())
