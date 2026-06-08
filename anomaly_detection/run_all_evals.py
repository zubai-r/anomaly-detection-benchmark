import os
import subprocess
import sys
import numpy as np

# List of all 15 categories in MVTec AD
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", 
    "hazelnut", "leather", "metal_nut", "pill", "screw", 
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

BACKBONES = ["resnet50", "dinov2", "dinov3"]

def run_evaluation(category, backbone):
    cmd = [
        "python", "-m", "anomaly_detection.eval",
        "--category", category,
        "--backbone", backbone
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running evaluation for category {category} with backbone {backbone}:")
        print(result.stderr)
        return False
    return True

def parse_result_file(category, backbone):
    log_file = os.path.join("results", f"eval_{category}_{backbone}.txt")
    if not os.path.exists(log_file):
        return None
        
    metrics = {}
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if "AUROC" in key or "F1-Max" in key or "AUPRO" in key:
                    metrics[key] = float(val.replace("%", ""))
                elif "Latency" in key:
                    metrics[key] = float(val.split()[0])
    return metrics

def main():
    os.makedirs("results", exist_ok=True)
    
    print("Starting comparative benchmarking on all 15 categories...")
    
    # Run evaluations
    for backbone in BACKBONES:
        print(f"\n========================================")
        print(f"Evaluating Backbone: {backbone}")
        print(f"========================================")
        for category in MVTEC_CATEGORIES:
            print(f"Evaluating Category: {category}...")
            # Run eval
            success = run_evaluation(category, backbone)
            if not success:
                print(f"Skipping {category} with {backbone} due to error.")
                
    # Collect results
    print("\nCollating results...")
    results = {bb: {} for bb in BACKBONES}
    
    for backbone in BACKBONES:
        for category in MVTEC_CATEGORIES:
            metrics = parse_result_file(category, backbone)
            if metrics:
                results[backbone][category] = metrics
                
    # Write collated results report
    report_path = os.path.join("results", "collated_results.md")
    
    with open(report_path, "w") as f:
        f.write("# MVTec AD Comparative Benchmark Report\n\n")
        f.write("This report summarizes the performance of three backbones (supervised ResNet-50, SSL DINOv2, and SSL DINOv3) under a training-free PatchCore-style pipeline.\n\n")
        
        # 1. Summary table of Averages
        f.write("## 1. Summary of Averages\n\n")
        f.write("| Backbone | Image-level AUROC | Image-level F1-Max | Pixel-level AUROC | Pixel-level AUPRO | Latency (ms/img) |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        
        summary_lines = []
        for bb in BACKBONES:
            img_aurocs = [results[bb][cat]["Image-level AUROC"] for cat in results[bb] if "Image-level AUROC" in results[bb][cat]]
            img_f1s = [results[bb][cat]["Image-level F1-Max"] for cat in results[bb] if "Image-level F1-Max" in results[bb][cat]]
            px_aurocs = [results[bb][cat]["Pixel-level AUROC"] for cat in results[bb] if "Pixel-level AUROC" in results[bb][cat]]
            px_aupros = [results[bb][cat]["Pixel-level AUPRO"] for cat in results[bb] if "Pixel-level AUPRO" in results[bb][cat]]
            latencies = [results[bb][cat]["Inference Latency"] for cat in results[bb] if "Inference Latency" in results[bb][cat]]
            
            avg_img_auroc = np.mean(img_aurocs) if img_aurocs else 0.0
            avg_img_f1 = np.mean(img_f1s) if img_f1s else 0.0
            avg_px_auroc = np.mean(px_aurocs) if px_aurocs else 0.0
            avg_px_aupro = np.mean(px_aupros) if px_aupros else 0.0
            avg_latency = np.mean(latencies) if latencies else 0.0
            
            f.write(f"| {bb} | {avg_img_auroc:.2f}% | {avg_img_f1:.2f}% | {avg_px_auroc:.2f}% | {avg_px_aupro:.2f}% | {avg_latency:.2f} ms |\n")
            summary_lines.append(f"{bb}: Im-AUROC={avg_img_auroc:.2f}%, Im-F1={avg_img_f1:.2f}%, Px-AUROC={avg_px_auroc:.2f}%, Px-AUPRO={avg_px_aupro:.2f}%, Latency={avg_latency:.2f}ms")
            
        f.write("\n")
        
        # 2. Detailed Per-Category Tables
        f.write("## 2. Per-Category Results\n\n")
        
        for metric_name in ["Image-level AUROC", "Pixel-level AUROC", "Pixel-level AUPRO", "Inference Latency"]:
            f.write(f"### {metric_name}\n\n")
            f.write("| Category | ResNet-50 | DINOv2 | DINOv3 |\n")
            f.write("| --- | --- | --- | --- |\n")
            
            for cat in MVTEC_CATEGORIES:
                vals = []
                for bb in BACKBONES:
                    if cat in results[bb] and metric_name in results[bb][cat]:
                        val = results[bb][cat][metric_name]
                        if "Latency" in metric_name:
                            vals.append(f"{val:.2f} ms")
                        else:
                            vals.append(f"{val:.2f}%")
                    else:
                        vals.append("N/A")
                f.write(f"| {cat} | {' | '.join(vals)} |\n")
            f.write("\n")
            
    print("\n========================================")
    print("COLLATED SUMMARY OF RESULTS:")
    print("========================================")
    for line in summary_lines:
        print(line)
    print("========================================")
    print(f"Saved collated report to: {report_path}")

if __name__ == "__main__":
    main()
