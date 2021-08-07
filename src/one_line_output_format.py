import warnings
from stable_baselines3.common.logger import FormatUnsupportedError, HumanOutputFormat, Video
from typing import Dict, TextIO, Union


class OneLineOutputFormat(HumanOutputFormat):
    def __init__(self, filename_or_file: Union[str, TextIO]):
        super().__init__(filename_or_file)

    def write(self, key_values: Dict, key_excluded: Dict, step: int=0) -> None:
        # Create strings for printing
        key2str = {}
        tag = None
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if excluded is not None and ("stdout" in excluded or "log" in excluded):
                continue

            if isinstance(value, Video):
                raise FormatUnsupportedError(["stdout", "log"], "video")

            if isinstance(value, float):
                # Align left
                value_str = f"{value:<8.3g}"
            else:
                value_str = str(value)

            if key.find("/") > 0:  # Find tag and add it to the dict
                tag = key[: key.find("/") + 1]
                #key2str[self._truncate(tag)] = ""
            # Remove tag from key
            if tag is not None and tag in key:
                key = str(key[len(tag) :])

            key2str[self._truncate(key)] = self._truncate(value_str)

        # Find max widths
        if len(key2str) == 0:
            warnings.warn("Tried to write empty key-value dict")
            return
        else:
            key_width = max(map(len, key2str.keys()))
            val_width = max(map(len, key2str.values()))

        line = f"Rollout {key2str['iterations']}: "

        for key in ['ep_rew_mean', 'val_set_forces', 'loss', 'policy_gradient_loss', 'value_loss']:
            if key in key2str:
                line += f"{key}: {key2str[key]}, "
        
        self.file.write(line[:-2] + '\n')   # Delete trailing comma
        self.file.flush()
