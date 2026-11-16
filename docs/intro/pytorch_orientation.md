# Orientation for PyTorch users

OpenSoundscape uses PyTorch "under the hood" for machine learning tasks. If you're familiar with PyTorch, these connections to the OpenSoundscape API should let you start using pieces of OpenSoundscape within your current workflows without much hacking:

- `AudioFileDataset` subclasss torch.dataset and include preprocessing and augmentation. If you use them as a drop-in substitution for a PyTorch Dataset, pay attention to your DataLoader's `collate_fn`. You can set the `collate_fn` to `opensoundscape.ml.dataloaders.collate_audio_samples_to_tensors` to stack samples and labels with a leading batch dimension, as is typical in PyTorch. 

- `opensoundscape.ml.CNN` class's `.network` attribute is a PyTorch model object.
- The CNN class implements high-level methods `.train()`, `.predict()`, `.eval()`, and `.generate_cams()` APIs which do what you expect them to
- `CNN.train()` and `.predict()` accept logger objects with a `.log()` method; subclass `opensoundscape.BaseLogger` to add setup and finish hooks. W&B runs are supported directly.

- the `CNN`'s `.preprocessor` attribute (which becomes AudioFileDataset's `.preprocessor` during `.train()` and `.predict()`) defines a sequential series of preprocessing & augmentation actions performed on each sample. Specifically, preprocessor.pipeline is a pd.Series and each item is an instance of the `opensoundscape.preprocess.action.Action` class (or subclass). See the preprocessing tutorial notebook for specifics on modifying preprocessing.
