from peft.tuners.tuners_utils import BaseTunerLayer
from typing import List, Any, Optional, Type
from .condition import condition_dict

class enable_lora:
    def __init__(self, lora_modules: List[BaseTunerLayer], activated: bool) -> None:
        self.activated: bool = activated
        if activated:
            return
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.scales = [
            {
                active_adapter: lora_module.scaling[active_adapter]
                for active_adapter in lora_module.active_adapters
            }
            for lora_module in self.lora_modules
        ]

    def __enter__(self) -> None:
        if self.activated:
            return

        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                if active_adapter in condition_dict.keys():
                    lora_module.scaling[active_adapter] = 0.0

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if self.activated:
            return
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]


class set_lora_scale:
    def __init__(self, lora_modules: List[BaseTunerLayer], scale: float) -> None:
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.scales = [
            {
                active_adapter: lora_module.scaling[active_adapter]
                for active_adapter in lora_module.active_adapters
            }
            for lora_module in self.lora_modules
        ]
        self.scale = scale

    def __enter__(self) -> None:
        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            lora_module.scale_layer(self.scale)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]
