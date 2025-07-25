#!/usr/bin/env python3
"""
简单的checkpoint推理脚本
可以选择在验证集或测试集上进行推理，完全复制训练时的validation逻辑
"""

import os
import torch
import json
import argparse
from sklearn.metrics import f1_score
import torch.nn.functional as F

# 导入项目模块
from main_classify import Model
from utils.datasets import create_loaders
import tqdm


def save_detailed_results(validation_step_outputs, save_dir, filename):
    """保存详细的验证结果到JSONL文件，完全复制原始训练时的逻辑"""
    try:
        out_path = os.path.join(save_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            for step_out in validation_step_outputs:
                image_paths = step_out.get('image_paths', None)
                texts = step_out['text']
                preds = step_out['pred_label']
                # probs = step_out['probs']
                labels = step_out['gt_label']
                
                # 处理批次中的每个样本
                batch_size = len(texts)
                for i in range(batch_size):
                    record = {
                        'image': image_paths[i] if image_paths is not None else None,
                        'text': texts[i],
                        'pred': preds[i].tolist() if isinstance(preds[i], torch.Tensor) else preds[i],
                        # 'prob': probs[i].tolist() if isinstance(probs[i], torch.Tensor) else probs[i],
                        'label': labels[i].tolist() if isinstance(labels[i], torch.Tensor) else labels[i],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"详细验证结果已保存到: {out_path}")
        return out_path
    except Exception as e:
        print(f"保存详细验证结果失败: {e}")
        return None

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """从checkpoint加载模型"""
    if not os.path.exists(checkpoint_path):
        print(f"错误: Checkpoint文件不存在: {checkpoint_path}")
        return None
    
    try:
        print("正在加载模型...")
        model = Model.load_from_checkpoint(
            checkpoint_path, 
            map_location=device, 
            strict=False
        )
        model = model.to(device)
        model.eval()
        print("模型加载成功!")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None


def run_inference(model, data_loader, device='cuda', dataset_type='food', save_dir=None, use_test=False):
    """运行推理，完全复制原始validation_step的逻辑"""
    # 处理tqdm包装的data_loader
    if hasattr(data_loader, 'iterable'):
        dataset_size = len(data_loader.iterable.dataset)
    else:
        dataset_size = len(data_loader.dataset)
    
    print(f"开始推理，数据集大小: {dataset_size} 样本")
    
    # 存储所有validation step的输出（完全复制原始逻辑）
    validation_step_outputs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                # 完全按照原始validation_step的逻辑
                text_input = batch["text"]
                img_input = batch["image"].to(device)
                gt_label = batch["label"].to(device)
                
                # 获取image_paths（如果有的话）
                image_paths = batch.get('image_path', None)
                
                # 计算模型输出（完全按照原始逻辑）
                outputs, extra_out = model.classifier(img_input, text_input)
                
                # 计算概率和预测（完全按照原始逻辑）
                if dataset_type in ["imdb", "rocov2"]:
                    probs = torch.sigmoid(outputs)
                    pred_label = (probs > 0.35).int()
                elif dataset_type in ["food", "snli", "chestxray"]:
                    probs = F.softmax(outputs, dim=1)
                    pred_label = torch.argmax(outputs, dim=1)
                else:
                    # 默认处理方式
                    probs = F.softmax(outputs, dim=1)
                    pred_label = torch.argmax(outputs, dim=1)
                
                # 计算损失（完全按照原始逻辑）
                loss_val = model.loss(outputs, gt_label.squeeze())
                
                # 构建返回字典（完全按照原始逻辑）
                ret_dict = {
                    'image_paths': image_paths,
                    'text': text_input,
                    'pred_label': pred_label.detach().cpu(),
                    'probs': probs.detach().cpu(),
                    'gt_label': gt_label.squeeze().detach().cpu(),
                    'loss': loss_val.detach().cpu(),
                }
                
                if extra_out is not None:
                    if "moe_scores" in extra_out:
                        ret_dict["moe_scores"] = extra_out["moe_scores"]
                    if "cls_" in extra_out:
                        ret_dict["cls"] = extra_out["cls_"].detach().cpu()
                
                validation_step_outputs.append(ret_dict)
                
            except Exception as e:
                print(f"处理批次 {batch_idx} 时出错: {e}")
                continue
    
    # 计算最终指标（完全按照原始on_validation_epoch_end的逻辑）
    if dataset_type == "food" or dataset_type == "snli" or dataset_type == "chestxray":
        all_cnt = 0
        correct_cnt = 0
        for step_out in validation_step_outputs:
            pred_label = step_out["pred_label"]
            gt_label = step_out["gt_label"]
            all_cnt += pred_label.size(0)
            correct_cnt += torch.sum(pred_label == gt_label).item()
        
        acc = correct_cnt / all_cnt
        print(f"\n=== 最终结果 ===")
        print(f"准确率 (Accuracy): {acc:.4f} ({acc*100:.2f}%)")
        
        # 保存详细验证结果到JSONL文件（完全按照原始逻辑）
        if save_dir:
            filename = "inference_details_test.jsonl" if use_test else "inference_details_val.jsonl"
            save_detailed_results(validation_step_outputs, save_dir, filename)
        
        return {"accuracy": acc}
        
    elif dataset_type == "imdb" or dataset_type == "rocov2":
        all_preds = []
        all_labels = []
        
        for step_out in validation_step_outputs:
            pred_label = step_out["pred_label"]
            gt_label = step_out["gt_label"]
            all_preds.append(pred_label)
            all_labels.append(gt_label)
        
        # 计算F1分数
        f1_macro = f1_score(
            torch.cat(all_labels).squeeze().cpu().numpy(),
            torch.cat(all_preds).cpu().numpy(),
            average="macro",
        )
        f1_micro = f1_score(
            torch.cat(all_labels).squeeze().cpu().numpy(),
            torch.cat(all_preds).cpu().numpy(),
            average="micro",
        )
        f1_samples = f1_score(
            torch.cat(all_labels).squeeze().cpu().numpy(),
            torch.cat(all_preds).cpu().numpy(),
            average="samples",
        )
        f1_weighted = f1_score(
            torch.cat(all_labels).squeeze().cpu().numpy(),
            torch.cat(all_preds).cpu().numpy(),
            average="weighted",
        )
        
        # 计算exact match accuracy
        preds_tensor = torch.cat(all_preds)
        labels_tensor = torch.cat(all_labels)
        exact_matches = (preds_tensor == labels_tensor).all(dim=1)
        val_acc = exact_matches.float().mean().item()
        
        print(f"\n=== 最终结果 ===")
        print(f"准确率 (Exact Match): {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Micro: {f1_micro:.4f}")
        print(f"F1 Samples: {f1_samples:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        
        # 保存详细验证结果到JSONL文件（完全按照原始逻辑）
        if save_dir:
            filename = "inference_details_test.jsonl" if use_test else "inference_details_val.jsonl"
            save_detailed_results(validation_step_outputs, save_dir, filename)
        
        return {
            "accuracy": val_acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_samples": f1_samples,
            "f1_weighted": f1_weighted
        }


def main():
    parser = argparse.ArgumentParser(description='从checkpoint进行推理')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint文件路径')
    parser.add_argument('--dataset', type=str, default='food', 
                       choices=['food', 'imdb', 'snli', 'chestxray', 'rocov2'],
                       help='数据集类型')
    parser.add_argument('--data_root', type=str, default='data/food-101', help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=32, help='batch大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--use_test', action='store_true', 
                       help='使用测试集而不是验证集（默认使用验证集，与训练时保持一致）')
    
    args = parser.parse_args()
    
    # 根据数据集类型设置正确的数据路径（与main_classify.py保持一致）
    if args.dataset == "food":
        data_path = "data/food-101"
    elif args.dataset == "imdb":
        data_path = "data/mmimdb"
    elif args.dataset == "snli":
        data_path = "data/snli-ve/data"  # SNLI特殊路径配置
    elif args.dataset == "chestxray":
        data_path = "data/chestXRay"
    elif args.dataset == "rocov2":
        data_path = "data/ROCOv2-radiology-main"
    else:
        data_path = args.data_root  # 回退到用户指定的路径
    
    # 如果用户没有明确指定data_root，使用配置的路径
    if args.data_root == 'data/food-101':  # 默认值，说明用户没有明确指定
        args.data_root = data_path
    
    print("=== Checkpoint推理脚本 ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"数据集: {args.dataset}")
    print(f"数据根目录: {args.data_root}")
    print(f"使用数据: {'测试集' if args.use_test else '验证集（与训练时一致）'}")
    print(f"Batch大小: {args.batch_size}")
    print(f"设备: {args.device}")
    
    # 加载模型
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    if model is None:
        return
    
    # 创建数据加载器
    print("\n正在创建数据加载器...")
    try:
        train_loader, val_loader, test_loader = create_loaders(
            args.data_root, 
            args.batch_size, 
            num_workers=4, 
            n_shot=0, 
            backbone="swinb_224"
        )
        
        # 选择使用的数据加载器
        eval_loader = test_loader if args.use_test else val_loader
        loader_name = "测试集" if args.use_test else "验证集"
        
        print(f"{loader_name}大小: {len(eval_loader.dataset)} 样本")
        print(f"批次数量: {len(eval_loader)}")
        
    except Exception as e:
        print(f"创建数据加载器失败: {e}")
        return
    
    # 运行推理
    print(f"\n开始在{loader_name}上推理...")
    #使用tqdm显示验证进度，显示进度条
    eval_loader_with_progress = tqdm.tqdm(eval_loader, desc="推理进度", unit="批次")
    
    # 获取保存目录（checkpoint所在的version目录）
    checkpoint_dir = os.path.dirname(args.checkpoint)
    
    results = run_inference(model, eval_loader_with_progress, args.device, args.dataset, save_dir=checkpoint_dir, use_test=args.use_test)
    
    # 保存结果
    checkpoint_dir = os.path.dirname(args.checkpoint)
    results_file = os.path.join(checkpoint_dir, f'inference_results_{"test" if args.use_test else "val"}.json')
    
    # 获取数据集大小
    if hasattr(eval_loader_with_progress, 'iterable'):
        dataset_size = len(eval_loader_with_progress.iterable.dataset)
    else:
        dataset_size = len(eval_loader.dataset)
    
    results_data = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'data_root': args.data_root,
        'use_test': args.use_test,
        'eval_type': 'test' if args.use_test else 'validation',
        'results': results,
        'sample_count': dataset_size
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")


if __name__ == "__main__":
    main()
