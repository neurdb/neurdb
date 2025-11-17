#!/usr/bin/env python3
"""
Trails: Evolutionary Algorithm for Model Selection

This script implements an evolutionary algorithm (EA) for neural architecture search:
1. Build a model population and score them using principled proxy (ExpressFlow)
2. Use EA to search towards models with highest proxy scores
3. Add diversity by splitting models into small/medium/large based on model size
4. Run separate EA for each size group, keep top 3 from each (total 9 models)
5. Use successive halving to select the best model
6. Train and test the final model
7. Record timing for model selection, training, and inference

Based on:
- neurdbrt.model.trails.proxies.expressflow
- neurdbrt.model.trails.search_space
- neurdbrt.model.trails.search_algorithm
"""

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import L1Loss, BCEWithLogitsLoss
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error, roc_auc_score
import torch_frame.data

from neurdbrt.model.trails.search_space import TrailsMLP, TrailsResNet
from neurdbrt.model.trails.proxies.expressflow import express_flow_score
from neurdbrt.model.trails.search_algorithm import evolutionary_algorithm
from relbench.base import TaskType

from neurdbrt.model.trails.utils.data import TableData


class SimpleLogger:
    """Simple logger wrapper that supports info, warning, success, section, banner, error methods"""
    
    def __init__(self, name: str = "trails", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Create console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def success(self, msg: str):
        self.logger.info(f"✓ {msg}")
    
    def section(self, msg: str):
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"  {msg}")
        self.logger.info("=" * 60)
    
    def banner(self, title: str, subtitle: str = "", details: str = ""):
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"  {title}")
        if subtitle:
            self.logger.info(f"  {subtitle}")
        if details:
            for line in details.split('\n'):
                if line.strip():
                    self.logger.info(f"  {line}")
        self.logger.info("=" * 70)
        self.logger.info("")


# Initialize logger
logger = SimpleLogger(name="trails", level="info")


default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
    torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
    torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
    torch_frame.multicategorical: (
        torch_frame.nn.MultiCategoricalEmbeddingEncoder,
        {},
    ),
    torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
    torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
}


def construct_stype_encoder_dict(
        stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any],
) -> Dict[torch_frame.stype, torch.nn.Module]:
    stype_encoder_dict = {
        stype: stype_encoder_cls_kwargs[stype][0](
            **stype_encoder_cls_kwargs[stype][1]
        )
        for stype in stype_encoder_cls_kwargs.keys()
    }
    return stype_encoder_dict


def deactivate_dropout(net: torch.nn.Module):
    """Deactivate dropout layers in the model for regression task"""
    deactive_nn_instances = (
        torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)
    for module in net.modules():
        if isinstance(module, deactive_nn_instances):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def test(net: torch.nn.Module, loader: torch.utils.data.DataLoader, early_stop: int = -1, is_regression: bool = False):
    """Test function for model evaluation"""
    pred_list = []
    y_list = []
    early_stop = early_stop if early_stop > 0 else len(loader.dataset)

    if not is_regression:
        net.eval()

    for idx, batch in tqdm(enumerate(loader), total=len(loader), leave=False, desc="Testing"):
        with torch.no_grad():
            batch = batch.to(device)
            y = batch.y.float()
            pred = net(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
            y_list.append(y.detach().cpu())
        if idx > early_stop:
            break
    pred_list = torch.cat(pred_list, dim=0)
    pred_logits = pred_list
    pred_list = torch.sigmoid(pred_list)
    y_list = torch.cat(y_list, dim=0).numpy()
    return pred_logits.numpy(), pred_list.numpy(), y_list


def create_evaluation_function(
        sample_batch_x: torch.Tensor,
        table_data: TableData,
        col_stats: Dict,
        col_names_dict: Dict,
        stype_encoder_dict: Dict,
        out_channels: int,
        space_name: str,
):
    """
    Create an evaluation function with bound parameters

    Args:
        space_instance: Instance of TrailsMLP or TrailsResNet
        sample_batch_x: encoded features
        table_data: TableData object
        col_stats: column statistics
        col_names_dict: column names dictionary
        stype_encoder_dict: stype encoder dictionary
        out_channels: output channels
        space_name: 'mlp' or 'resnet'

    Returns:
        Callable function that takes architecture and returns score
    """

    def evaluate_func(arch: List[int]) -> float:
        # Dynamically build model based on arch parameter
        try:
            num_cols = get_num_cols(table_data)
            if space_name == 'mlp':
                # Create MLP with specific architecture
                model = TrailsMLP(
                    channels=num_cols,
                    out_channels=out_channels,
                    num_layers=len(arch) + 1,
                    col_stats=col_stats,
                    col_names_dict=col_names_dict,
                    stype_encoder_dict=stype_encoder_dict,
                    hidden_dims=arch,
                    normalization='layer_norm',
                    dropout_prob=0.2,
                ).to(device)
                net_for_proxy = model.mlp
            else:  # resnet
                # Create ResNet with specific architecture
                model = TrailsResNet(
                    channels=num_cols,
                    out_channels=out_channels,
                    num_layers=len(arch),
                    col_stats=col_stats,
                    col_names_dict=col_names_dict,
                    stype_encoder_dict=stype_encoder_dict,
                    block_widths=arch,
                    normalization='layer_norm',
                    dropout_prob=0.2,
                ).to(device)
                net_for_proxy = model.backbone

            # Compute ExpressFlow score
            score, _ = express_flow_score(
                arch=net_for_proxy,
                batch_data=sample_batch_x,
                device=str(device),
                use_wo_embedding=False,
                linearize_target=None,
                epsilon=1e-5,
                weight_mode="traj_width",
                use_fp64=False,
            )

            # Clean up model
            del model

        except Exception as e:
            logger.warning(f"Error computing proxy for arch {arch}: {e}")
            score = -1e10  # Very low score for failed architectures
        finally:
            # Clean up after each evaluation
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        return float(score)

    return evaluate_func


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_cols(table_data):
    """Get the number of columns from table_data"""
    # Try to get from col_stats first
    if 'num_cols' in table_data.col_stats:
        return table_data.col_stats['num_cols']

    # If not available, calculate from the first batch
    sample_batch = next(iter(torch_frame.data.DataLoader(table_data.train_tf, batch_size=1, shuffle=False)))
    if hasattr(sample_batch, 'x'):
        return sample_batch.x.shape[1]
    else:
        # Sum up all feature dimensions
        total_dims = sum(feat.shape[1] for feat in sample_batch.feat_dict.values())
        return total_dims


def prepare_sample_batch_for_proxy(
        table_data: TableData,
        space_name: str,
        sample_size: int = 256,
) -> torch.Tensor:
    """
    Prepare sample batch for proxy evaluation

    Args:
        table_data: TableData object
        space_name: 'mlp' or 'resnet'
        device: device to use
        sample_size: number of samples to use

    Returns:
        Encoded features tensor
    """
    logger.info(f"Preparing sample batch for proxy evaluation...")
    sample_size = min(sample_size, len(table_data.train_tf))
    sample_indices = random.sample(range(len(table_data.train_tf)), sample_size)
    sample_subset = Subset(table_data.train_tf, sample_indices)
    sample_loader = torch_frame.data.DataLoader(sample_subset, batch_size=min(4, sample_size), shuffle=False)

    # Get one batch and encode it
    batch = next(iter(sample_loader)).to(device)

    # Create encoder
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    out_channels = 1

    # Create temporary model for encoding
    # Get the number of columns
    num_cols = get_num_cols(table_data)

    if space_name == 'mlp':
        temp_model = TrailsMLP(
            channels=num_cols,
            out_channels=out_channels,
            num_layers=2,
            col_stats=table_data.col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            hidden_dims=[num_cols],
            normalization='layer_norm',
            dropout_prob=0.2,
        ).to(device)
    else:
        temp_model = TrailsResNet(
            channels=num_cols,
            out_channels=out_channels,
            num_layers=2,
            col_stats=table_data.col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            block_widths=[num_cols, num_cols],
            normalization='layer_norm',
            dropout_prob=0.2,
        ).to(device)

    # Encode features
    with torch.no_grad():
        x_encoded, _ = temp_model.encoder(batch)
        if space_name == 'mlp':
            x_encoded = torch.mean(x_encoded, dim=1)
        else:
            x_encoded = x_encoded.view(x_encoded.size(0), -1)

    del temp_model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logger.success(f"Encoded features: {x_encoded.shape}")
    return x_encoded


def calculate_and_group_architectures(
        space_name: str,
        output_file: str = None,
) -> Tuple[Dict, float]:
    """
    Calculate model capacities and group architectures by size
    Generate ALL possible architectures (no sampling)
    Uses global disk file - dataset independent!

    Args:
        space_name: 'mlp' or 'resnet'
        output_file: Optional file to save results

    Returns:
        Tuple of (grouped_architectures_dict, calculation_time)
    """
    logger.section("Calculating Model Capacities and Grouping")
    logger.info(f"   Space: {space_name}")
    logger.info(f"   Mode: ALL possible architectures (dataset-independent)")

    # Try to load existing global results first
    global_dir = "./result_raw_from_server/hyperopt_sh_baseline"
    os.makedirs(global_dir, exist_ok=True)
    global_file = os.path.join(global_dir, f"capacity_groups_{space_name}_global.json")
    if os.path.exists(global_file):
        logger.info(f"Loading global results from {global_file}...")
        try:
            with open(global_file, 'r') as f:
                grouped_architectures = json.load(f)
            # Convert string representations back to lists
            for group in ['small', 'medium', 'large']:
                if group in grouped_architectures:
                    grouped_architectures[group] = [eval(arch) if isinstance(arch, str) else arch
                                                    for arch in grouped_architectures[group]]
            logger.success(f"Loaded global results: {global_file}")
            logger.info(f"   Small group: {len(grouped_architectures.get('small', []))} architectures")
            logger.info(f"   Medium group: {len(grouped_architectures.get('medium', []))} architectures")
            logger.info(f"   Large group: {len(grouped_architectures.get('large', []))} architectures")
            return grouped_architectures, 0.0  # No calculation time for loading
        except Exception as e:
            logger.warning(f"Failed to load global results: {e}")
            logger.info(f"Will recalculate...")

    # If no global results exist, calculate from scratch
    logger.info(f"Generating ALL possible architectures...")
    start_time = time.time()
    architectures_with_capacity = []

    # Generate ALL possible architectures (no sampling needed)
    all_channels = [32, 64, 128, 256]  # Define available channels
    num_blocks_range = (2, 4)  # Define block range

    logger.info(f"   Channels: {all_channels}")
    logger.info(f"   Block range: {num_blocks_range}")

    # Calculate total number of possible architectures
    total_possible = 0
    for num_blocks in range(num_blocks_range[0], num_blocks_range[1] + 1):
        total_possible += len(all_channels) ** num_blocks

    logger.info(f"   Total possible architectures: {total_possible}")

    # Generate ALL possible architectures
    for num_blocks in range(num_blocks_range[0], num_blocks_range[1] + 1):
        logger.info(f"   Processing {num_blocks}-block architectures...")

        # Generate all combinations for this number of blocks
        from itertools import product
        for arch in product(all_channels, repeat=num_blocks):
            arch = list(arch)

            # Calculate capacity using minimal model (dataset-independent)
            if space_name == 'mlp':
                model = TrailsMLP(
                    channels=10,  # Minimal input size
                    out_channels=1,
                    num_layers=len(arch) + 1,
                    col_stats={'num_cols': 10},  # Minimal stats
                    col_names_dict={},
                    stype_encoder_dict={},
                    hidden_dims=arch,
                    normalization='layer_norm',
                    dropout_prob=0.2,
                )
            else:  # resnet
                model = TrailsResNet(
                    channels=10,  # Minimal input size
                    out_channels=1,
                    num_layers=len(arch),
                    col_stats={'num_cols': 10},  # Minimal stats
                    col_names_dict={},
                    stype_encoder_dict={},
                    block_widths=arch,
                    normalization='layer_norm',
                    dropout_prob=0.2,
                )

            # Calculate capacity
            capacity = model.estimate_capacity(include_bias=True)

            # Clean up
            del model

            architectures_with_capacity.append((arch, capacity))

    logger.success(f"Generated {len(architectures_with_capacity)} architectures")

    # Step 2: Sort by capacity and group
    logger.info(f"Sorting and grouping architectures...")
    architectures_with_capacity.sort(key=lambda x: x[1])
    total_archs = len(architectures_with_capacity)

    # Define size groups based on capacity percentiles
    small_threshold = architectures_with_capacity[total_archs // 3][1]
    large_threshold = architectures_with_capacity[2 * total_archs // 3][1]

    logger.info(f"   Small threshold: < {small_threshold}")
    logger.info(f"   Large threshold: >= {large_threshold}")

    # Group architectures by size
    small_archs = [arch for arch, cap in architectures_with_capacity if cap < small_threshold]
    medium_archs = [arch for arch, cap in architectures_with_capacity if small_threshold <= cap < large_threshold]
    large_archs = [arch for arch, cap in architectures_with_capacity if cap >= large_threshold]

    logger.info(f"   Small group: {len(small_archs)} architectures")
    logger.info(f"   Medium group: {len(medium_archs)} architectures")
    logger.info(f"   Large group: {len(large_archs)} architectures")

    # Prepare results
    grouped_architectures = {
        'small': small_archs,
        'medium': medium_archs,
        'large': large_archs,
        'thresholds': {
            'small_threshold': small_threshold,
            'large_threshold': large_threshold
        },
        'statistics': {
            'total_architectures': total_archs,
            'small_count': len(small_archs),
            'medium_count': len(medium_archs),
            'large_count': len(large_archs)
        }
    }

    calculation_time = time.time() - start_time
    logger.success(f"Capacity calculation complete: {calculation_time:.2f}s")

    # Save results to global file
    logger.info(f"Saving global results to {global_file}...")
    with open(global_file, 'w') as f:
        json.dump(grouped_architectures, f, indent=2)
    logger.success(f"Global results saved to {global_file}")

    # Also save to specified output file if different
    if output_file and output_file != global_file:
        logger.info(f"Copying to {output_file}...")
        if output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(grouped_architectures, f, indent=2)
        elif output_file.endswith('.csv'):
            import pandas as pd
            # Create a DataFrame with all architectures and their groups
            data = []
            for group_name, archs in [('small', small_archs), ('medium', medium_archs), ('large', large_archs)]:
                for arch in archs:
                    data.append({
                        'architecture': str(arch),
                        'group': group_name,
                        'num_layers': len(arch),
                        'total_channels': sum(arch)
                    })
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
        logger.success(f"Results also saved to {output_file}")

    return grouped_architectures, calculation_time


def diversity_based_selection(
        space_name: str,
        table_data: TableData,
        sample_batch_x: torch.Tensor,
        col_stats: Dict,
        col_names_dict: Dict,
        stype_encoder_dict: Dict,
        out_channels: int,
        models_per_size: int = 5,
) -> List[Tuple[List[int], float, str, float]]:
    logger.section("Diversity-Based Selection")
    logger.info(f"   Pre-calculating capacities and grouping by size")
    logger.info(f"   Keeping top {models_per_size} from each size group")

    # Step 1: Calculate capacities and group architectures (dataset-independent!)
    grouped_architectures, capacity_time = calculate_and_group_architectures(
        space_name=space_name,
        output_file=f"capacity_groups_{space_name}_global.json",
    )

    logger.info(f"   Capacity calculation time: {capacity_time:.2f}s")

    # Extract grouped architectures
    small_archs = grouped_architectures['small']
    medium_archs = grouped_architectures['medium']
    large_archs = grouped_architectures['large']

    all_results = []

    # Step 2: Run EA for each size group
    size_groups = [
        ('small', small_archs),
        ('medium', medium_archs),
        ('large', large_archs)
    ]

    for size_group, group_architectures in size_groups:
        if len(group_architectures) == 0:
            logger.warning(f"   Skipping {size_group} group (no architectures)")
            continue

        logger.info(f"Running EA for {size_group} models...")

        # Create evaluation function
        evaluate_func = create_evaluation_function(
            sample_batch_x=sample_batch_x,
            table_data=table_data,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            out_channels=out_channels,
            space_name=space_name,
        )

        # Run EA for this size group with constrained search space
        # Get the model class for EA
        if space_name == 'mlp':
            model_class = TrailsMLP
        else:  # resnet
            model_class = TrailsResNet
        ea_results = evolutionary_algorithm(
            model_class=model_class,
            evaluate_func=evaluate_func,
            population_size=20,  # Smaller population for each group
            generations=5,  # Fewer generations for each group
            elite_size=5,
            mutation_rate=0.3,
            # Constrain search to architectures from this size group
            allowed_architectures=group_architectures,
        )

        # Keep top models from this size group
        ea_results.sort(key=lambda x: x[1], reverse=True)
        top_models = ea_results[:models_per_size]

        # Add size group info to results (with placeholder val_score for consistency)
        for arch, score in top_models:
            all_results.append((arch, score, size_group, None))  # None as placeholder for val_score

        logger.info(f"     Found {len(ea_results)} models in {size_group} group")
        logger.info(f"     Selected top {len(top_models)} models")
        if top_models:
            logger.info(f"     Best {size_group} score: {top_models[0][1]:.4f}")
            logger.info(f"     {size_group.upper()} group selected models:")
            for i, (arch, score) in enumerate(top_models):
                logger.info(f"       {i+1}. {arch} (score: {score:.4f})")

        # Clean up space_instance and force garbage collection
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force garbage collection
        import gc
        gc.collect()

    logger.info(f"Total selected models: {len(all_results)}")
    logger.info(f"   Target: {models_per_size} × 3 = {models_per_size * 3}")

    # Remove duplicates based on architecture (keep the one with highest score)
    logger.info(f"Removing duplicates...")
    unique_results = {}
    for arch, score, group, val_score in all_results:
        arch_key = tuple(arch)  # Convert to tuple for hashing
        if arch_key not in unique_results or score > unique_results[arch_key][1]:
            unique_results[arch_key] = (arch, score, group, val_score)

    # Convert back to list and sort by score
    deduplicated_results = list(unique_results.values())
    deduplicated_results.sort(key=lambda x: x[1], reverse=True)  # Sort by proxy score

    logger.success(f"After deduplication: {len(deduplicated_results)} unique models")

    # Print summary of all selected models by group
    logger.section("FINAL SELECTED MODELS SUMMARY")
    for size_group, group_architectures in [('small', small_archs), ('medium', medium_archs), ('large', large_archs)]:
        group_models = [(arch, score, group, val_score) for arch, score, group, val_score in deduplicated_results if group == size_group]
        logger.info(f"{size_group.upper()} GROUP ({len(group_models)} models):")
        for i, (arch, score, group, val_score) in enumerate(group_models):
            logger.info(f"     {i+1}. {arch} (proxy_score: {score:.4f})")

    return deduplicated_results


def successive_halving(
        selected_models: List[Tuple[List[int], float, str, float]],
        space_name: str,
        table_data: TableData,
        is_regression: bool,
        max_epochs: int = 50,
        min_epochs: int = 1,
) -> Tuple[List[int], float]:
    logger.section("Successive Halving Selection")
    logger.info(f"   Candidates: {len(selected_models)}")

    # Prepare data loaders (use original train/val/test splits)
    # Use normal batch size for training
    train_loader = torch_frame.data.DataLoader(table_data.train_tf, batch_size=256, shuffle=True)
    val_loader = torch_frame.data.DataLoader(table_data.val_tf, batch_size=256, shuffle=False)

    # Successive halving
    candidates = selected_models.copy()
    current_epochs = min_epochs

    while len(candidates) > 1 and current_epochs <= max_epochs:
        logger.info(f"   Round: {len(candidates)} candidates, {current_epochs} epochs")

        # Train each candidate for current_epochs
        candidate_scores = []

        for i, (arch, proxy_score, size_group, val_score) in enumerate(candidates):
            logger.info(f"     Training candidate {i + 1}/{len(candidates)}: {arch} ({size_group})")

            # Create model
            stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
            out_channels = 1

            num_cols = get_num_cols(table_data)
            if space_name == 'mlp':
                model = TrailsMLP(
                    channels=num_cols,
                    out_channels=out_channels,
                    num_layers=len(arch) + 1,
                    col_stats=table_data.col_stats,
                    col_names_dict=table_data.col_names_dict,
                    stype_encoder_dict=stype_encoder_dict,
                    hidden_dims=arch,
                    normalization='layer_norm',
                    dropout_prob=0.2,
                ).to(device)
            else:  # resnet
                model = TrailsResNet(
                    channels=num_cols,
                    out_channels=out_channels,
                    num_layers=len(arch),
                    col_stats=table_data.col_stats,
                    col_names_dict=table_data.col_names_dict,
                    stype_encoder_dict=stype_encoder_dict,
                    block_widths=arch,
                    normalization='layer_norm',
                    dropout_prob=0.2,
                ).to(device)

            # Train model (always train in successive halving)
            model, _ = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                is_regression=is_regression,
                num_epochs=current_epochs,
                lr=0.001,
                max_batches_per_epoch=20,
                early_stop_patience=10,
            )

            # Evaluate on validation set
            val_score = evaluate_model(
                model=model,
                test_loader=val_loader,
                is_regression=is_regression,
            )[0]  # Get metric, ignore inference time

            candidate_scores.append((arch, proxy_score, size_group, val_score))

            # Clean up
            del model
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Force garbage collection
            import gc
            gc.collect()

        # Sort by validation score
        if is_regression:
            candidate_scores.sort(key=lambda x: x[3])  # Lower MAE is better
        else:
            candidate_scores.sort(key=lambda x: x[3], reverse=True)  # Higher AUC is better

        # Keep top half
        keep_count = max(1, len(candidates) // 2)
        candidates = candidate_scores[:keep_count]

        logger.info(f"     Kept top {len(candidates)} candidates")
        if candidates:
            best_score = candidates[0][3]
            logger.info(f"     Best validation score: {best_score:.4f}")

        # Increase epochs for next round
        if current_epochs < max_epochs:
            current_epochs = min(max_epochs, current_epochs * 2)
        else:
            # If we've reached max_epochs, we're done
            break

    # Return best architecture and its score
    if candidates:
        best_arch, _, best_size_group, best_val_score = candidates[0]
        logger.success(f"Best model: {best_arch} ({best_size_group})")
        if best_val_score is not None:
            logger.info(f"   Validation score: {best_val_score:.4f}")
        else:
            logger.info(f"   Validation score: None (using proxy score)")
        return best_arch, best_val_score
    else:
        raise ValueError("No candidates remaining after successive halving")


def train_model(
        model: torch.nn.Module,
        train_loader: torch_frame.data.DataLoader,
        val_loader: torch_frame.data.DataLoader,
        is_regression: bool,
        num_epochs: int = 200,
        lr: float = 0.001,
        max_batches_per_epoch: int = 20,
        early_stop_patience: int = 10,
) -> Tuple[torch.nn.Module, float]:
    """Train model using the training logic from train_final_model"""

    logger.section("TRAINING MODEL")
    logger.info(f"Training configuration:")
    logger.info(f"  num_epochs: {num_epochs}")
    logger.info(f"  early_stop_threshold: {early_stop_patience}")
    logger.info(f"  batch_size: 256")
    logger.info(f"  lr: {lr}")
    logger.info(f"  max_round_epoch: {max_batches_per_epoch}")

    train_start = time.time()

    try:
        # Setup loss and optimizer
        if is_regression:
            loss_fn = L1Loss()
            deactivate_dropout(model)
            higher_is_better = False
        else:
            loss_fn = BCEWithLogitsLoss()
            higher_is_better = True

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # Setup data loaders
        data_loaders = {
            "train": train_loader,
            "val": val_loader,
        }

        model.to(device)
        patience = 0
        best_val_metric = -math.inf if higher_is_better else math.inf
        best_model_state = None

        # Training loop
        logger.info("Training...")

        for epoch in range(num_epochs):
            model.train()
            loss_accum = 0
            count_accum = 0

            for idx, batch in enumerate(data_loaders["train"]):
                if idx > max_batches_per_epoch:
                    break

                # Clear cache before each batch
                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()

                optimizer.zero_grad()
                batch = batch.to(device)
                pred = model(batch)
                pred = pred.view(-1) if pred.size(1) == 1 else pred
                y = batch.y.float()
                loss = loss_fn(pred, y)

                loss.backward()
                optimizer.step()
                loss_accum += loss.item()
                count_accum += 1

                # Clear cache after each batch
                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()

            # Validation
            val_logits, _, val_pred_hat = test(
                model, data_loaders["val"], is_regression=is_regression)

            # Clear cache after validation
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()

            # Calculate metric
            if is_regression:
                # For regression, use raw logits (not sigmoided values)
                val_metric = mean_absolute_error(val_logits, val_pred_hat)
            else:
                if len(np.unique(val_pred_hat)) > 1:
                    val_metric = roc_auc_score(val_pred_hat, val_logits)
                else:
                    val_metric = 0.5

            # Early stopping
            if (higher_is_better and val_metric > best_val_metric) or \
                    (not higher_is_better and val_metric < best_val_metric):
                best_val_metric = val_metric
                best_model_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience > early_stop_patience:
                    logger.info(f"  Early stopped at epoch {epoch}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch + 1}: val_metric={val_metric:.4f}")

        # Training ends here
        train_end = time.time()
        train_time_seconds = train_end - train_start

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        logger.success(f"Training completed!")
        logger.info(f"   Best validation metric: {best_val_metric:.6f}")
        logger.info(f"   Training time: {train_time_seconds:.2f} seconds ({train_time_seconds / 3600:.2f} hours)")

        return model, train_time_seconds

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def evaluate_model(
        model: torch.nn.Module,
        test_loader: torch_frame.data.DataLoader,
        is_regression: bool,
) -> Tuple[float, float]:
    """Evaluate model on test data"""
    start_time = time.time()
    test_logits, test_pred_hat, test_y = test(model, test_loader, is_regression=is_regression)

    inference_time = time.time() - start_time

    # Calculate metric
    if is_regression:
        # For regression, use raw logits (not sigmoided values)
        metric = mean_absolute_error(test_y, test_logits)
    else:
        if len(np.unique(test_y)) > 1:
            metric = roc_auc_score(test_y, test_pred_hat)
        else:
            metric = 0.5

    return metric, inference_time


def main():
    parser = argparse.ArgumentParser(description='Trails: Evolutionary Algorithm for Model Selection')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--space_name', type=str, required=True, choices=['mlp', 'resnet'], help='Search space')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Update global device variable for imported functions
    global device
    device = torch.device(args.device)

    set_seed(args.seed)

    logger.banner(
        "TRAILS",
        "Evolutionary Algorithm for Model Selection",
        f"Data: {args.data_dir}\nSpace: {args.space_name}\nDevice: {args.device}\nOutput: {args.output_csv}"
    )

    # Load data
    logger.info(f"Loading data...")
    table_data = TableData.load_from_dir(args.data_dir)

    # Determine task type
    if table_data.task_type == TaskType.REGRESSION:
        is_regression = True
    else:
        is_regression = False
    logger.info(f"Task type: {table_data.task_type}")

    # Prepare sample batch for proxy evaluation
    x_encoded = prepare_sample_batch_for_proxy(
        table_data=table_data,
        space_name=args.space_name,
        sample_size=256,
    )

    # Create encoder and output channels
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    out_channels = 1

    # Step 1: Diversity-based selection with EA
    logger.section("Step 1: Diversity-based selection with EA")
    selection_start_time = time.time()

    selected_models = diversity_based_selection(
        space_name=args.space_name,
        table_data=table_data,
        sample_batch_x=x_encoded,
        col_stats=table_data.col_stats,
        col_names_dict=table_data.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
        out_channels=out_channels,
        models_per_size=5,
    )

    selection_time = time.time() - selection_start_time
    logger.success(f"Model selection complete: {selection_time:.2f}s")

    # Step 2: Successive halving
    logger.section("Step 2: Successive halving")

    best_arch, best_val_score = successive_halving(
        selected_models=selected_models,
        space_name=args.space_name,
        table_data=table_data,
        is_regression=is_regression,
        max_epochs=50,
        min_epochs=1,
    )

    logger.success(f"Best architecture: {best_arch}")
    logger.info(f"   Validation score: {best_val_score:.4f}")

    # Step 3: Train final model
    logger.section("Step 3: Training final model")
    train_start_time = time.time()

    # Create final model
    num_cols = get_num_cols(table_data)
    if args.space_name == 'mlp':
        final_model = TrailsMLP(
            channels=num_cols,
            out_channels=out_channels,
            num_layers=len(best_arch) + 1,
            col_stats=table_data.col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            hidden_dims=best_arch,
            normalization='layer_norm',
            dropout_prob=0.2,
        ).to(args.device)
    else:
        final_model = TrailsResNet(
            channels=num_cols,
            out_channels=out_channels,
            num_layers=len(best_arch),
            col_stats=table_data.col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            block_widths=best_arch,
            normalization='layer_norm',
            dropout_prob=0.2,
        ).to(args.device)

    # Deactivate dropout for regression
    if is_regression:
        deactivate_dropout(final_model)

    # Train final model (use original train/val splits)
    train_loader = torch_frame.data.DataLoader(table_data.train_tf, batch_size=256, shuffle=True)
    val_loader = torch_frame.data.DataLoader(table_data.val_tf, batch_size=256, shuffle=False)

    final_model, train_time = train_model(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        is_regression=is_regression,
        num_epochs=200,
        lr=0.001,
        max_batches_per_epoch=20,
        early_stop_patience=10,
    )

    logger.success(f"Final training complete: {train_time:.2f}s")

    # Step 4: Test final model
    logger.section("Step 4: Testing final model")

    test_loader = torch_frame.data.DataLoader(table_data.test_tf, batch_size=256, shuffle=False)

    test_metric, inference_time = evaluate_model(
        model=final_model,
        test_loader=test_loader,
        is_regression=is_regression,
    )

    logger.success(f"Testing complete: {test_metric:.4f}, {inference_time:.2f}s")

    # Calculate total time
    total_time = selection_time + train_time + inference_time

    # Save results to CSV
    logger.info(f"Saving results to CSV...")

    # Check if CSV exists
    csv_exists = os.path.exists(args.output_csv)

    # Extract data source from data_dir
    data_source = os.path.basename(os.path.dirname(args.data_dir))

    # Prepare result row
    result_row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': os.path.basename(args.data_dir),
        'data_source': data_source,
        'architecture': args.space_name,
        'selection_time_seconds': selection_time,
        'final_train_time_seconds': train_time,
        'inference_time_seconds': inference_time,
        'total_time_seconds': total_time,
        'best_val_metric': best_val_score,
        'final_best_val_metric': best_val_score,
        'final_test_metric': test_metric,
        'best_params': str(best_arch),
        'metric': 'mae' if is_regression else 'roc_auc',
    }

    # Write to CSV
    with open(args.output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result_row.keys())
        if not csv_exists:
            writer.writeheader()
        writer.writerow(result_row)

    logger.success(f"Results saved to: {args.output_csv}")

    # Print summary
    logger.section("Trails Complete!")
    logger.info(f"Results:")
    logger.info(f"   Best architecture: {best_arch}")
    logger.info(f"   Test metric: {test_metric:.4f}")
    logger.info(f"   Selection time: {selection_time:.2f}s")
    logger.info(f"   Training time: {train_time:.2f}s")
    logger.info(f"   Inference time: {inference_time:.2f}s")
    logger.info(f"   Total time: {total_time:.2f}s")
    logger.info(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
