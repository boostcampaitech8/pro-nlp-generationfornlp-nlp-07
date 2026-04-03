"""Training callbacks"""

from pathlib import Path
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import shutil


class SaveBestModelCallback(TrainerCallback):
    """
    Callback to save the best model to a separate directory during training
    """
    
    def __init__(self, best_model_dir: str = None):
        """
        Args:
            best_model_dir: Directory to save the best model (default: {output_dir}/best_model)
        """
        self.best_model_dir = best_model_dir
        self.best_metric = None
        self.best_checkpoint = None
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called after evaluation"""
        # Get the metric used for best model selection
        metric_for_best = args.metric_for_best_model if hasattr(args, 'metric_for_best_model') else 'eval_loss'
        greater_is_better = args.greater_is_better if hasattr(args, 'greater_is_better') else False
        
        # Get current metric value
        if state.log_history:
            last_log = state.log_history[-1]
            current_metric = last_log.get(metric_for_best)
            
            if current_metric is not None:
                # Check if this is the best model so far
                is_best = False
                if self.best_metric is None:
                    is_best = True
                elif greater_is_better:
                    is_best = current_metric > self.best_metric
                else:
                    is_best = current_metric < self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
                    self.best_checkpoint = state.global_step
                    
                    # Save best model
                    if self.best_model_dir is None:
                        self.best_model_dir = str(Path(args.output_dir) / "best_model")
                    
                    best_model_path = Path(self.best_model_dir)
                    best_model_path.mkdir(parents=True, exist_ok=True)
                    
                    # Find the checkpoint directory
                    checkpoint_dir = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
                    
                    if checkpoint_dir.exists():
                        # Copy checkpoint to best_model directory
                        print(f"\n🎉 New best model found! (step {state.global_step}, {metric_for_best}={current_metric:.4f})")
                        print(f"Saving best model to {best_model_path}")
                        
                        # Remove old best model if exists
                        if best_model_path.exists():
                            shutil.rmtree(best_model_path)
                        
                        # Copy current checkpoint to best_model
                        shutil.copytree(checkpoint_dir, best_model_path)
                        
                        # Save original checkpoint information
                        checkpoint_info = {
                            "original_checkpoint": checkpoint_dir.name,
                            "step": state.global_step,
                            "metric": current_metric,
                            "metric_name": metric_for_best
                        }
                        import json
                        info_path = best_model_path / "best_model_info.json"
                        info_path.write_text(json.dumps(checkpoint_info, indent=2))
                        
                        print(f"Best model saved to {best_model_path}")
                        print(f"Original checkpoint: {checkpoint_dir.name} (step {state.global_step})")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training"""
        if self.best_checkpoint is not None:
            print(f"\n✅ Training completed. Best model was at step {self.best_checkpoint} "
                  f"with metric {self.best_metric:.4f}")
            print(f"Best model saved to: {self.best_model_dir}")

