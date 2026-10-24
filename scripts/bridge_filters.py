"""Trajectory filters for the course BridgeData V2 subset."""

from __future__ import annotations

import tensorflow as tf


def _instruction(trajectory: dict) -> tf.Tensor:
    """Return one normalized instruction from an RLDS trajectory."""
    language = tf.reshape(trajectory["language_instruction"], [-1])[0]
    return tf.strings.lower(tf.strings.strip(language))


def course_task_filter_v2(trajectory: dict) -> tf.Tensor:
    """Keep placement trajectories related to the two Task-B skills.

    The two groups are object-on-surface and object-in-container. Other skills
    remain available in the downloaded shard trial but are not used to train.
    """
    text = _instruction(trajectory)
    padded = tf.strings.join([" ", text, " "])
    on_relation = tf.strings.regex_full_match(
        padded, r".*( on | onto | on top of ).*"
    )
    surface_target = tf.strings.regex_full_match(
        text, r".*(towel|cloth|plate|tray|table|surface).*"
    )
    in_relation = tf.strings.regex_full_match(
        padded, r".*( in | into | inside ).*"
    )
    container_target = tf.strings.regex_full_match(
        text, r".*(basket|bowl|pot|pan|container|bin|cup).*"
    )
    on_surface = tf.logical_and(on_relation, surface_target)
    in_container = tf.logical_and(in_relation, container_target)
    return tf.logical_or(on_surface, in_container)


# Retain the old public name for manifest tools while the versioned name makes
# Octo create a fresh dataset-statistics cache after this filter correction.
course_task_filter = course_task_filter_v2


def classify_instruction(text: str) -> str:
    """Python equivalent used for manifests and reports."""
    normalized = " ".join(text.lower().strip().split())
    surfaces = ("towel", "cloth", "plate", "tray", "table", "surface")
    containers = ("basket", "bowl", "pot", "pan", "container", "bin", "cup")
    if any(token in normalized for token in (" on ", " onto ", "on top of")) and any(
        target in normalized for target in surfaces
    ):
        return "on_surface"
    if any(token in normalized for token in (" in ", " into ", " inside ")) and any(
        target in normalized for target in containers
    ):
        return "in_container"
    return "other"
