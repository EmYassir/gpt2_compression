import os
import logging
import copy  

from ray import tune, init
from ray.tune import CLIReporter
from ray.tune.schedulers import (
    ASHAScheduler, 
    HyperBandForBOHB, 
    MedianStoppingRule, 
    PopulationBasedTraining
)
from transformers_local.trainer_utils import (
    BestRun,
    EvalPrediction,
    set_seed,
    PREFIX_CHECKPOINT_DIR
)
from typing import Dict, Any

logger = logging.getLogger(__name__)

def default_compute_objective(metrics: Dict[str, float]) -> float:
    metrics = copy.deepcopy(metrics)
    loss = metrics.pop("eval_loss", None)
    _, _ = metrics.pop("epoch", None), metrics.pop("loss", None)
    # Remove speed metrics
    speed_metrics = [m for m in metrics.keys() if m.endswith("_runtime") or m.endswith("_per_second")]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
    return loss if len(metrics) == 0 else sum(metrics.values())

def modified_compute_objective(metrics: Dict[str, float]) -> float:
    metrics = copy.deepcopy(metrics)
    _ = metrics.pop("eval_loss", None)
    #if loss is not None:
    #    return loss
    acc_and_f1, em_and_f1 = metrics.pop("acc_and_f1", None), metrics.pop("em_and_f1", None)
    if acc_and_f1 is not None:
        return acc_and_f1
    elif em_and_f1 is not None:
        return em_and_f1
    acc, em = metrics.pop("acc", None), metrics.pop("em", None)
    if acc is not None:
        return acc
    elif em is not None:
        return em
    # Sum up the rest of metrics
    _, _ = metrics.pop("epoch", None), metrics.pop("loss", None)
    speed_metrics = [m for m in metrics.keys() if m.endswith("_runtime") or m.endswith("_per_second")]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
    logger.info(f'Objective == sum')
    return sum(metrics.values())

def hp_search_setup(args, trial: Dict[str, Any]):
    """HP search setup code"""
    for key, value in trial.items():
        if not hasattr(args, key):
            raise AttributeError(
                f"Trying to set {key} in the hyperparameter search but there is no corresponding field in `TrainingArguments`."
            )
        old_attr = getattr(args, key, None)
        # Casting value to the proper type
        if old_attr is not None:
            value = type(old_attr)(value)
        setattr(args, key, value)


def tune_save_checkpoint(trainer, global_step):
    with tune.checkpoint_dir(step = global_step) as checkpoint_dir:
        output_dir = os.path.join(trainer.args.output_dir, checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)


def ray_hp_space_pkd_minilm(trial = None) -> Dict[str, float]:
    return {
        "temperature": tune.choice([1., 2., 5.]),                                               # 3 values
        "alpha_ce": tune.grid_search([0.0, 0.1, 0.2,  0.5, 0.7, 1.0]),                          # 6 values
        "alpha_att": tune.grid_search([0.0, 0.1, 0.2,  0.5, 0.7, 1.0]),                         # 6 values
        "alpha_val": tune.grid_search([0.0, 0.1, 0.2,  0.5, 0.7, 1.0]),                         # 6 values
        "alpha_pkd": tune.grid_search([0.0, 0.1, 0.2, 0.5, 0.7, 1.0]),                          # 6 values
        "beta_pkd": tune.grid_search([0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 100.]),                     # 7 values
        #"num_train_epochs": tune.choice(list(range(28, 31))),                                   # 3 values
                                                                                                # 27,216 possible combinations in total
    }
"""
def ray_hp_space_pkd_minilm(trial = None) -> Dict[str, float]:
    return {
        "temperature": tune.choice([1, 2]),                                               
        "alpha_ce": tune.grid_search([0.2, 0.5]),                                        
        "alpha_att": tune.grid_search([ 0.2, 0.5]),                                       
        "alpha_val": tune.grid_search([0.2, 0.5]),                                       
        "alpha_pkd": tune.grid_search([ 0.2, 0.5]),                                      
        "beta_pkd": tune.grid_search([0.2, 0.5]),                                 
        "num_train_epochs": tune.choice(list(range(4, 5))),                          
                                                                                               
    }
"""

def ray_hp_space_default(trial = None) -> Dict[str, float]:
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "num_train_epochs": tune.choice(list(range(3, 7))),
        "seed": tune.uniform(1, 40),
    }

def report_to_hp_search(
    trainer, trial: Dict[str, Any], epoch: int, metrics: Dict[str, float]
):
    if trial is None:
        return
    trainer.objective = trainer.compute_objective(metrics.copy())
    tune.report(objective=trainer.objective, **metrics)

def run_hp_search_ray(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    def _objective(trial, local_trainer, checkpoint_dir=None):
        checkpoint = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    checkpoint = os.path.join(checkpoint_dir, subdir)
        local_trainer.objective = None
        global_step, _ = local_trainer.train(resume_from_checkpoint=checkpoint, trial=trial) 
        metrics, _, _ = local_trainer.evaluate(use_tqdm=False)
        local_trainer.objective = local_trainer.compute_objective(metrics)
        #tune_save_checkpoint(local_trainer, global_step)
        tune.report(objective=local_trainer.objective, **metrics, done=True)

    # Setup default `resources_per_trial`.
    if "resources_per_trial" not in kwargs:
        # Default to 1 CPU and 1 GPU (if applicable) per trial.
        kwargs["resources_per_trial"] = {"cpu": 4}
        if trainer.args.n_gpu > 0:
            kwargs["resources_per_trial"]["gpu"] = 1
        resource_msg = "4 CPU" + (" and 1 GPU" if trainer.args.n_gpu > 0 else "")
        logger.info(
            "No `resources_per_trial` arg was passed into "
            "`hyperparameter_search`. Setting it to a default value "
            f"of {resource_msg} for each trial."
        )
    # Make sure each trainer only uses GPUs that were allocated per trial.
    gpus_per_trial = kwargs["resources_per_trial"].get("gpu", 0)
    trainer.args.n_gpu = gpus_per_trial

    # Setup default `progress_reporter`.
    if "progress_reporter" not in kwargs:
        kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])
    if "keep_checkpoints_num" in kwargs and kwargs["keep_checkpoints_num"] > 0:
        trainer.uses_tune_checkpoints = True
        if kwargs["keep_checkpoints_num"] > 1:
            logger.warning(
                f"Currently keeping {kwargs['keep_checkpoint_num']} checkpoints for each trial. "
                "Checkpoints are usually huge, "
                "consider setting `keep_checkpoints_num=1`."
            )
    if "scheduler" in kwargs:
        # Check if checkpointing is enabled for PopulationBasedTraining
        if isinstance(kwargs["scheduler"], PopulationBasedTraining):
            if not trainer.use_tune_checkpoints:
                logger.warning(
                    "You are using PopulationBasedTraining but you haven't enabled checkpointing. "
                    "This means your trials will train from scratch everytime they are exploiting "
                    "new configurations. Consider enabling checkpointing by passing "
                    "`keep_checkpoints_num=1` as an additional argument to `Trainer.hyperparameter_search`."
                )

        # Check for `do_eval` and `eval_during_training` for schedulers that require intermediate reporting.
        if isinstance(
            kwargs["scheduler"], (ASHAScheduler, MedianStoppingRule, HyperBandForBOHB, PopulationBasedTraining)
        ) and (not trainer.args.do_eval or trainer.args.evaluation_strategy == IntervalStrategy.NO):
            raise RuntimeError(
                "You are using {cls} as a scheduler but you haven't enabled evaluation during training. "
                "This means your trials will not report intermediate results to Ray Tune, and "
                "can thus not be stopped early or used to exploit other trials parameters. "
                "If this is what you want, do not use {cls}. If you would like to use {cls}, "
                "make sure you pass `do_eval=True` and `evaluation_strategy='steps'` in the "
                "Trainer `args`.".format(cls=type(kwargs["scheduler"]).__name__)
            )
    init(include_dashboard=False)
    analysis = tune.run(
        tune.with_parameters(_objective, local_trainer=trainer),
        config=trainer.hp_space(None),
        num_samples=n_trials,
        **kwargs,
    )
    best_trial = analysis.get_best_trial(metric="objective", mode=direction[:3])
    best_config = analysis.get_best_config(metric="objective", mode=direction[:3])

    logger.info(f"############ Best trial: {best_trial.config}")
    logger.info(f"############ Best config: {best_config}")
    best_run = BestRun(best_trial.trial_id, best_trial.last_result["objective"], best_trial.config)
    return best_run