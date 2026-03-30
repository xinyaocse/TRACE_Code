import subprocess
import os
import sys


def check_file_exists(filepath):
    return os.path.exists(filepath)


def run_ablation_experiments(dataset_name, backbone, target_cat, query_img=""):
    print(f"\n{'=' * 60}")
    print(f"Running Ablation Studies: {dataset_name} with {backbone}")
    print(f"{'=' * 60}")

    target_dir = f"./target_images_{dataset_name}_{backbone}"
    iae_dir = f"./IAE_outputs_{dataset_name}_{backbone}"

    ablation_dir_no_iae = f"./Ablation_{dataset_name}_{backbone}_noIAE"
    adv_no_iae = os.path.join(ablation_dir_no_iae, "adv_noIAE.png")

    if check_file_exists(adv_no_iae):
        print(f"\n[SKIP] Ablation (no IAE) already done: {adv_no_iae}")
    else:
        print(f"\n[Ablation 1] Running without IAE augmentation...")
        cmd = [
            "python", "ablation.py",
            "--dataset", dataset_name,
            "--model", backbone,
            "--target_imgs_dir", target_dir,
            "--IAE_path", iae_dir,
            "--substitute_dir", "./checkpoints",
            "--outputpath", ablation_dir_no_iae,
            "--max_iter", "200",
            "--no_IAE"
        ]
        subprocess.run(cmd, check=True)
    ablation_dir_no_rie = f"./Ablation_{dataset_name}_{backbone}_noRIE"
    adv_no_rie = os.path.join(ablation_dir_no_rie, "adv_noRIE.png")

    if check_file_exists(adv_no_rie):
        print(f"\n[SKIP] Ablation (no RIE) already done: {adv_no_rie}")
    else:
        print(f"\n[Ablation 2] Running without RIE module...")
        cmd = [
            "python", "ablation.py",
            "--dataset", dataset_name,
            "--model", backbone,
            "--target_imgs_dir", target_dir,
            "--IAE_path", iae_dir,
            "--substitute_dir", "./checkpoints",
            "--outputpath", ablation_dir_no_rie,
            "--max_iter", "200",
            "--no_RIE"
        ]
        subprocess.run(cmd, check=True)
    print(f"\n[Ablation Evaluation] Evaluating all ablation results...")
    cmd_eval = [
        "python", "evaluate.py",
        "--dataset", dataset_name,
        "--model", backbone,
        "--k", "10",
        "--target_label", str(target_cat),
        "--eval_ablation"
    ]
    subprocess.run(cmd_eval, check=True)


def run_dataset_pipeline(
        dataset_name,
        backbone,
        bit=64,
        csq_epochs=30,
        csq_bs=64,
        sub_epochs=20,
        z=4,
        k=15,
        target_cat=7,
        m=50,
        query_img="",
        run_ablation=True
):

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}, Backbone: {backbone}")
    print(f"{'=' * 60}")


    victim_ckpt = f"./csq_models/csq_{dataset_name}_{backbone}_{bit}.pth"
    if check_file_exists(victim_ckpt):
        print(f"\n[SKIP] CSQ model already exists: {victim_ckpt}")
    else:
        print(f"\n[Step 1] Training CSQ victim model...")
        cmd = [
            "python", "hashing/CSQ.py",
            "--dataset", dataset_name,
            "--backbone", backbone,
            "--bit", str(bit),
            "--epochs", str(csq_epochs),
            "--center_loss_w", "0.1",
            "--batch_size", str(csq_bs)
        ]
        subprocess.run(cmd, check=True)

    models_to_train = ["alexnet", "vgg16", "resnet50", "densenet121","ViT","Clip"]
    if backbone in models_to_train:
        models_to_train.remove(backbone)  # 排除受害模型

    all_exist = True
    for model in models_to_train:
        ckpt = f"./checkpoints/substitute_{dataset_name}_{model}.pth"
        if not check_file_exists(ckpt):
            all_exist = False
            break

    if all_exist:
        print(f"\n[SKIP] All substitute models already exist")
    else:
        print(f"\n[Step 2] Training substitute models...")
        cmd = [
            "python", "hashing/train_substitute_ensemble.py",
            "--dataset_name", dataset_name,
            "--victim_ckpt", victim_ckpt,
            "--backbone", backbone,
            "--out_dim", str(bit),
            "--epochs", str(sub_epochs),
            "--z", str(z),
            "--k", str(k)
        ]
        subprocess.run(cmd, check=True)

    iae_dir = f"./IAE_outputs_{dataset_name}_{backbone}"
    iae_0 = os.path.join(iae_dir, "IAE_0.png")

    if check_file_exists(iae_0):
        print(f"\n[SKIP] IAE augmentation already done: {iae_0}")
    else:
        print(f"\n[Step 3] Performing IAE target augmentation...")
        target_dir = f"./target_images_{dataset_name}_{backbone}"
        cmd = [
            "python", "IAE_augmentation.py",
            "--dataset", dataset_name,
            "--target_category", str(target_cat),
            "--m", str(m),
            "--substitute_dir", "./checkpoints",
            "--target_imgs_dir", target_dir,
            "--IAE_path", iae_dir,
            "--steps", "200",
            "--lr", "0.002"
        ]
        subprocess.run(cmd, check=True)

    TRACE_dir = f"./TRACE_outputs_{dataset_name}_{backbone}"
    adv_final = os.path.join(TRACE_dir, "adv_final.png")

    if check_file_exists(adv_final):
        print(f"\n[SKIP] TRACE attack already done: {adv_final}")
    else:
        print(f"\n[Step 4] Performing TRACE attack...")
        cmd = [
            "python", "TRACE_attack.py",
            "--dataset", dataset_name,
            "--model", backbone,
            "--IAE_path", iae_dir,
            "--outputpath", TRACE_dir,
            "--substitute_dir", "./checkpoints",
            "--max_iter", "200"
        ]
        subprocess.run(cmd, check=True)

    print(f"\n[Step 5] Evaluating TRACE attack performance...")
    cmd = [
        "python", "evaluate.py",
        "--dataset", dataset_name,
        "--model", backbone,
        "--adv_path", adv_final,
        "--k", "10",
        "--target_label", str(target_cat)
    ]
    subprocess.run(cmd, check=True)

    if run_ablation:
        run_ablation_experiments(dataset_name, backbone, target_cat, query_img)


def generate_summary_report():
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY REPORT")
    print("=" * 80)

    datasets = ["mnist", "cifar10", "oxford5k_db", "paris6k_db"]
    backbones = ["alexnet", "vgg16", "resnet50", "densenet121"]

    results = []

    for dataset in datasets:
        for backbone in backbones:
            # 检查结果文件
            TRACE_result = f"./TRACE_outputs_{dataset}_{backbone}/adv_final.png"
            no_iae_result = f"./Ablation_{dataset}_{backbone}_noIAE/adv_noIAE.png"
            no_rie_result = f"./Ablation_{dataset}_{backbone}_noRIE/adv_noRIE.png"

            status = []
            if os.path.exists(TRACE_result):
                status.append("TRACE✓")
            if os.path.exists(no_iae_result):
                status.append("NoIAE✓")
            if os.path.exists(no_rie_result):
                status.append("NoRIE✓")

            if status:
                results.append(f"{dataset}-{backbone}: {', '.join(status)}")
            else:
                results.append(f"{dataset}-{backbone}: Not completed")

    for result in results:
        print(result)

    print("\n" + "=" * 80)


def main():
    experiments = [
        {
            "dataset": "mnist",
            "target_cat": 7,
            "csq_epochs": 30,
            "csq_bs": 64,
            "sub_epochs": 20,
        },
        {
            "dataset": "cifar10",
            "target_cat": 3,
            "csq_epochs": 30,
            "csq_bs": 64,
            "sub_epochs": 20,
        },
        {
            "dataset": "oxford5k_db",
            "target_cat": 0,
            "csq_epochs": 20,
            "csq_bs": 16,
            "sub_epochs": 15,
        },
        {
            "dataset": "paris6k_db",
            "target_cat": 0,
            "csq_epochs": 20,
            "csq_bs": 16,
            "sub_epochs": 15,
        },
        {
            "dataset": "imagenet_db",
            "target_cat": 0,
            "csq_epochs": 20,
            "csq_bs": 16,
            "sub_epochs": 15,
        },
        {
            "dataset": "coco_db",
            "target_cat": 0,
            "csq_epochs": 20,
            "csq_bs": 16,
            "sub_epochs": 15,
        }
    ]

    backbones = ["alexnet", "vgg16", "resnet50", "densenet121","ViT","Clip"]

    run_ablation = True
    if "--no-ablation" in sys.argv:
        run_ablation = False
        print("Note: Ablation studies are disabled")

    for exp in experiments:
        for backbone in backbones:
            try:
                run_dataset_pipeline(
                    dataset_name=exp["dataset"],
                    backbone=backbone,
                    bit=64,
                    csq_epochs=exp["csq_epochs"],
                    csq_bs=exp["csq_bs"],
                    sub_epochs=exp["sub_epochs"],
                    z=4,
                    k=10,
                    target_cat=exp["target_cat"],
                    m=50,
                    query_img="",
                    run_ablation=run_ablation
                )
            except Exception as e:
                print(f"\nError in {exp['dataset']} with {backbone}: {e}")
                continue
    generate_summary_report()

    print("\nAll experiments completed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--clean":
        print("Cleaning previous results...")
        os.system("rm -rf ./csq_models/")
        os.system("rm -rf ./checkpoints/")
        os.system("rm -rf ./IAE_outputs*/")
        os.system("rm -rf ./TRACE_outputs*/")
        os.system("rm -rf ./Ablation*/")
        os.system("rm -rf ./substitute_data_images/")
        os.system("rm -rf ./query_results*.pt")
        os.system("rm -rf ./target_images*/")
        os.system("rm -rf ./auto_query*.png")
        print("Clean completed!")
        sys.exit(0)

    main()
