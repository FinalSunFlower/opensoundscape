"""Logging interfaces and helpers for exporting content."""

import inspect
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any
import warnings

import pandas as pd
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram


class BaseLogger(ABC):
    """Base class for logging metrics from OpenSoundscape models."""

    def setup(self, model=None, stage=None, **kwargs) -> None:
        """Prepare the logger for a model operation."""

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        """Log a mapping of metric names to values."""

    def finish(self, model=None, stage=None, **kwargs) -> None:
        """Finish a model operation."""


def _is_wandb_session(logger):
    config = getattr(logger, "config", None)
    return (
        callable(getattr(logger, "log", None))
        and callable(getattr(logger, "watch", None))
        and callable(getattr(config, "update", None))
    )


def _module_hook_mappings(model):
    if model is None or not hasattr(model, "network"):
        return []

    hook_attributes = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks_with_kwargs",
        "_forward_hooks_always_called",
    )
    mappings = []
    for module in model.network.modules():
        for attribute in hook_attributes:
            hooks = getattr(module, attribute, None)
            if isinstance(hooks, dict):
                mappings.append((module, attribute, set(hooks)))
        for parameter in module.parameters(recurse=False):
            hooks = getattr(parameter, "_backward_hooks", None)
            mappings.append(
                (
                    parameter,
                    "_backward_hooks",
                    set(hooks) if isinstance(hooks, dict) else set(),
                )
            )
    return mappings


def _new_hook_mappings(model, previous_mappings):
    previous = {
        (id(owner), attribute): hook_ids
        for owner, attribute, hook_ids in previous_mappings
    }
    return [
        (
            owner,
            attribute,
            hook_ids.difference(previous.get((id(owner), attribute), set())),
        )
        for owner, attribute, hook_ids in _module_hook_mappings(model)
        if hook_ids.difference(previous.get((id(owner), attribute), set()))
    ]


class _LoggerAdapter:
    def __init__(self, logger):
        self.logger = logger
        self.is_wandb = _is_wandb_session(logger)
        self._depth = 0
        self._model = None
        self._stage = None
        self._hook_mappings = []
        self._watch_requested = False

    def setup(self, model=None, stage=None):
        if self._depth:
            self._depth += 1
            return

        self._depth = 1
        self._model = model
        self._stage = stage
        if self.is_wandb:
            if hasattr(model, "_generate_wandb_config"):
                self.update_config(model._generate_wandb_config())
            if stage == "train" and hasattr(model, "wandb_logging"):
                log_freq = model.wandb_logging["watch_freq"]
                if log_freq is not None:
                    previous_mappings = _module_hook_mappings(model)
                    self._watch_requested = True
                    try:
                        self.logger.watch(
                            model.network,
                            log="all",
                            log_freq=log_freq,
                            log_graph=model.wandb_logging["log_graph"],
                        )
                    finally:
                        self._hook_mappings = _new_hook_mappings(
                            model, previous_mappings
                        )
        elif isinstance(self.logger, BaseLogger):
            self.logger.setup(model=model, stage=stage)

    def update_config(self, values):
        if self.is_wandb:
            self.logger.config.update(values)

    def log(self, metrics, step=None, **kwargs):
        if self.logger is None:
            return

        method = self.logger.log
        if self.is_wandb:
            method(metrics, **kwargs)
            return

        try:
            parameters = inspect.signature(method).parameters.values()
            accepts_step = any(
                parameter.name == "step"
                or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            accepts_step = False

        if accepts_step:
            method(metrics, step=step, **kwargs)
        else:
            payload = metrics if step is None else {**metrics, "step": step}
            method(payload)

    def _unwatch(self):
        if not self._watch_requested:
            return

        try:
            unwatch = getattr(self.logger, "unwatch", None)
            if callable(unwatch):
                try:
                    parameters = inspect.signature(unwatch).parameters.values()
                    accepts_model = any(
                        parameter.kind
                        in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.VAR_POSITIONAL,
                        )
                        for parameter in parameters
                    )
                except (TypeError, ValueError):
                    accepts_model = True
                if accepts_model:
                    unwatch(self._model.network)
                else:
                    unwatch()
        finally:
            for owner, attribute, hook_ids in self._hook_mappings:
                hooks = getattr(owner, attribute, None)
                if isinstance(hooks, dict):
                    for hook_id in hook_ids:
                        hooks.pop(hook_id, None)

    def finish(self, model=None, stage=None):
        if self._depth == 0:
            return

        self._depth -= 1
        if self._depth:
            return

        model = self._model if model is None else model
        stage = self._stage if stage is None else stage
        try:
            if self.is_wandb:
                self._unwatch()
            elif isinstance(self.logger, BaseLogger):
                self.logger.finish(model=model, stage=stage)
        finally:
            self._hook_mappings = []
            self._watch_requested = False
            self._model = None
            self._stage = None


def _resolve_logger(logger=None, wandb_session=None):
    if logger is not None and wandb_session is not None:
        raise ValueError("Pass either logger or wandb_session, not both")

    if wandb_session is not None:
        warnings.warn(
            "wandb_session is deprecated; pass it as logger instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        logger = wandb_session

    if isinstance(logger, _LoggerAdapter):
        return logger
    if logger is not None and not callable(getattr(logger, "log", None)):
        raise TypeError("logger must be None or provide a callable .log() method")
    return _LoggerAdapter(logger)


def logger_lifecycle(stage):
    """Wrap a model operation in logger setup and cleanup."""

    def decorate(method):
        signature = inspect.signature(method)

        @wraps(method)
        def wrapped(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            adapter = _resolve_logger(
                bound.arguments.get("logger"),
                bound.arguments.get("wandb_session"),
            )
            bound.arguments["logger"] = adapter
            if "wandb_session" in bound.arguments:
                bound.arguments["wandb_session"] = None

            try:
                adapter.setup(model=self, stage=stage)
                return method(*bound.args, **bound.kwargs)
            finally:
                adapter.finish(model=self, stage=stage)

        return wrapped

    return decorate


def wandb_table(
    dataset,
    n=None,
    classes_to_extract=(),
    random_state=None,
    raise_exceptions=False,
    drop_labels=False,
    gradcam_model=None,
):
    """Generate a wandb Table visualizing n random samples from a sample_df

    Args:
        dataset: object to generate samples, eg AudioFileDataset or AudioSplittingDataset
        n: number of samples to generate (randomly selected from df)
            - if None, does not subsample or change order
        bypass_augmentations: if True, augmentations in Preprocessor are skipped
        classes_to_extract: tuple of classes - will create columns containing the scores/labels
        random_state: default None; if integer provided, used for reproducible random sample
        drop_labels: if True, does not include 'label' column in Table
        gradcam_model: if not None, will generate GradCAMs for each sample using gradcam_model.get_cams()
            - requires optional dependency `pip install grad-cam`

    Returns: a W&B Table of preprocessed samples with labels and playable audio

    """
    import wandb

    # select which clips to generate
    if n is None or len(dataset) < n:
        # if not enough samples to make n, just use all of them (don't complain)
        pass
    else:
        # randomly choose entire files from the table of labels
        dataset = dataset.sample(n=n, random_state=random_state)

    # set up columns for WandB display table
    table_columns = [
        "audio",
        "tensor",
        "labels",
        "path",
        "clip start time",
        "clip duration",
    ]
    for c in classes_to_extract:
        table_columns.append(c)

    sample_table = pd.DataFrame(columns=table_columns)
    for i in range(len(dataset)):
        try:
            # generate the content for a new row of the table
            sample = dataset[i]
            audio = Audio.from_file(
                sample.source, offset=sample.start_time, duration=sample.duration
            )

            # use .data as image if image-like; otherwise make spec
            try:
                image = wandb.Image(sample.data * -1)
            except ValueError:
                # the sample is not image-like. Make a spectrogram
                array = Spectrogram.from_audio(audio).to_image(return_type="np")
                image = wandb.Image(array * -1)

            # add contents to this row of the table
            row_info = [
                wandb.Audio(audio.samples, audio.sample_rate),  # audio object
                image,  # spectrogram image
                sample.categorical_labels,
                str(sample.source),
                sample.start_time,
                sample.duration,
            ]
            for c in classes_to_extract:
                row_info += [sample.labels[c]]

            # add new row to wandb Table
            sample_table.loc[len(sample_table)] = row_info

        except:  # by default, ignore exceptions
            if raise_exceptions:
                raise

    # add GradCAMs to table
    if gradcam_model is not None:
        try:
            import pytorch_grad_cam

            grad_cam_available = True
        except ImportError as exc:
            # cannot generate CAMs without grad-cam dependency, but this is an optional feature so just warn and skip
            warnings.warn(
                "Generating GradCAMs for the wandb table requires the optional 'grad-cam' dependency. "
                "Install it with `pip install opensoundscape[grad-cam]` or `pip install grad-cam` to enable this feature."
            )
            grad_cam_available = False
        if grad_cam_available:
            for c in classes_to_extract:
                samples = dataset.label_df
                samples = gradcam_model.generate_cams(samples, classes=[c])
                cam_images = []
                for s in samples:
                    array = s.cam.create_rgb_heatmaps(class_subset=[c])
                    cam_images.append(wandb.Image(array))
                sample_table[f"{c} GradCAM"] = cam_images

    if drop_labels:
        sample_table = sample_table.drop(columns=["labels"])

    return wandb.Table(dataframe=sample_table)
