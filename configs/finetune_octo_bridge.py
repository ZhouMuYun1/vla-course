"""Course Task-B finetuning configuration for Octo-Small 1.5."""

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec


def get_config(config_string="head_only,language_conditioned"):
    mode, task = config_string.split(",")
    assert mode in ("head_only", "head_mlp_only", "full")
    assert task in ("language_conditioned", "image_conditioned", "multimodal")

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    else:
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )

    max_steps = FieldReference(10_000)
    window_size = FieldReference(2)
    dataset_kwargs = {
        "name": "bridge_dataset",
        "data_dir": placeholder(str),
        "image_obs_keys": {"primary": "image_0", "wrist": None},
        "proprio_obs_key": "proprio",
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        "action_normalization_mask": [True, True, True, True, True, True, False],
        "standardize_fn": ModuleSpec.create(
            "octo.data.oxe.oxe_standardization_transforms:bridge_dataset_transform"
        ),
        "filter_functions": [
            ModuleSpec.create("scripts.bridge_filters:course_task_filter_v2")
        ],
        "num_parallel_reads": 4,
        "num_parallel_calls": 8,
    }
    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=300000,
        batch_size=16,
        shuffle_buffer_size=4000,
        num_steps=max_steps,
        log_interval=50,
        eval_interval=1000,
        save_interval=2000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(project="octo_finetune", group="task_b", entity=None),
        dataset_kwargs=dataset_kwargs,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=1e-4,
                warmup_steps=500,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,
        ),
        val_kwargs=dict(val_shuffle_buffer_size=500, num_val_batches=8),
        viz_kwargs=dict(
            eval_batch_size=8,
            trajs_for_metrics=16,
            trajs_for_viz=4,
            samples_per_state=4,
        ),
        traj_transform_kwargs=dict(
            window_size=window_size,
            action_horizon=4,
            goal_relabeling_strategy=None,
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(keep_image_prob=0.0),
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256), "wrist": (128, 128)},
            image_augment_kwargs=dict(
                primary=dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
                wrist=dict(
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            ),
        ),
        frame_transform_threads=8,
    )
    return ConfigDict(config)
