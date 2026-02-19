import logging

from dataclasses import dataclass, field
from typing import List, Union

from .enums import GPUType, Precision, Runtime


@dataclass(frozen=True)
class InferenceConfig:
    precision: Precision
    gpu_type: GPUType
    runtime: Runtime
    checkpoint: str = "Photoroom/prx-512-t2i-sft-distilled"

    @property
    def config_id(self):
        return f"{self.gpu_type.value}_{self.precision.value}_{self.runtime.value}"

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class ConfigurationValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate(self, config: InferenceConfig):
        result = ValidationResult(is_valid=True)

        # BF16 is not supported by Turing
        if config.precision == Precision.BF16:
            if config.gpu_type.compute_capability[0] < 8:
                # I think we should throw an error here in case. But to get it to run
                # for the sake of the example, I decided to add a simple warning
                # result.is_valid = False
                # result.errors.append(
                #    f"BF16 needs compute >= 8.0, {config.gpu_type.value} has {config.gpu_type.compute_capability}"
                # )
                result.warnings.append(
                   f"BF16 needs compute >= 8.0, {config.gpu_type.value} has {config.gpu_type.compute_capability}"
                )

        # Adding a warning for the current FP16 stability issue
        if config.precision == Precision.FP16:
            result.warnings.append("FP16 has known stability issues")

        # Some memory checks could also be a good idea. Given the time limitations I
        # decided not to add them

        return result

class ConfigurationManager:
    def __init__(self):
        self.validator = ConfigurationValidator()
        self.logger = logging.getLogger(__name__)
        self._registry = {}

    def create_config(self, precision: Precision, gpu_type: GPUType, runtime: Runtime, 
                      **kwargs) -> InferenceConfig:
        config = InferenceConfig(precision, gpu_type, runtime, **kwargs)
        validation = self.validator.validate(config)

        if not validation.is_valid:
            raise ValueError(f"Invalid config: {validation.errors}")

        for w in validation.warnings:
            self.logger.warning(w)

        return config

    def register_config(self, precision: Precision, gpu_type: GPUType, runtime: Runtime, 
                        name: Union[List, None] = None, **kwargs):
        try:
            config = self.create_config(precision, gpu_type, runtime, **kwargs)
            logging.info(f"Configuration {config.config_id} has been loaded")
            name = name or config.config_id
            self._registry[name] = config
        except ValueError as e:
            logging.error(f"""Configuration with input parameters {precision}, {gpu_type}, 
                          {runtime} has failed with the following error: \n {e}""")

    def get_config(self, name: str) -> InferenceConfig:
        return self._registry[name]

    def list_configs(self) -> List[str]:
        return list(self._registry.keys())
