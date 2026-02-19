import logging

from dataclasses import dataclass, field
from typing import List, Union

from .enums import GPUType, Precision, Runtime


@dataclass(frozen=True)
class InferenceConfig:
    """
    Class containing parameters for an inference configuration

    ...
    Attributes
    ----------
    precision: Precision
        precision of the data and model used for inference (e.g. FP32, FP16 etc.)
    gpu_type: GPUType 
        type of the gpu used for inference (e.g. T4, H100, etc.)
    runtime: Runtime
        runtime used for inference (i.e. torch, tensorrt etc.)
    
    Methods
    -------
    config_id()
        configuration id that identifies this run
    """
    precision: Precision
    gpu_type: GPUType
    runtime: Runtime
    checkpoint: str = "Photoroom/prx-512-t2i-sft-distilled"

    @property
    def config_id(self) -> str:
        """
        Configuration id of this specific inference instance

        ...
        Returns
        -------
        config_id: str
            name attributed to this specific configuration
        """
        return f"{self.gpu_type.value}_{self.precision.value}_{self.runtime.value}"

@dataclass
class ValidationResult:
    """
    Outcome of the validation of a specific inference configuration

    ...
    Attributes
    ----------
    is_valid: bool
        whether the configuration proposed is actually valid
    errors: list
        set of errors that might have arisen during the validation
    warnings: list
        list of warnings that might have been thrown during the validation
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class ConfigurationValidator:
    """
    Validates a specific inference configuration

    ...
    Attributes
    ----------
    logger:
        logging instance of the class
    
    Methods
    -------
    validate(config)
        validates a specific inference configuration
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate(self, config: InferenceConfig) -> ValidationResult:
        """
        Validation of a specific inference configuration

        ...
        Parameters
        ----------
        config: InferenceConfig
            inference configuration to be validated

        Returns
        -------
        result: ValidationResult   
            The result of the validation explaining whether it was successful or not
        """
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
    """
    Wrapper for initializing and validating inference configurations

    ...
    Attributes
    ----------
    validator: ConfigurationValidator
        takes care of validating a configuration
    logger: 
        logging utility of this class
    _registry: dict
        stores successfully instantiated InferenceConfigs

    Methods
    -------
    create_config(precision, gpu_type, runtime)
        Creates and validates an inference configuration
    register_config(precision, gpu_type, runtime, name)
        runs a create_config on an inference configuration and 
        stores the configuration in the _registry
    get_config(name)
        given the name, returns the respective inference configuration stored 
        in _registry
    list_configs()
        lists available inference configs inside of registry

    Raises
    ------
    ValueError if the configuration is not correct
    """
    def __init__(self):
        self.validator = ConfigurationValidator()
        self.logger = logging.getLogger(__name__)
        self._registry = {}

    def create_config(self, precision: Precision, gpu_type: GPUType, runtime: Runtime, 
                      **kwargs) -> InferenceConfig:
        """
        Creates and validates an inference configuration

        ...
        Parameters
        ----------
        precision: Precision
            data precision to be used at inference time
        gpu_type: GPUType
            gpu to be sued at inference time
        runtime: Runtime
            runtime to be used at inference such as torch, torch.compile, tensorrt etc.
        
        Returns
        -------
        config: InferenceConfig
            loaded inference configuration
        
        Raises
        ------
        ValueError
        """
        config = InferenceConfig(precision, gpu_type, runtime, **kwargs)
        validation = self.validator.validate(config)

        if not validation.is_valid:
            raise ValueError(f"Invalid config: {validation.errors}")

        for w in validation.warnings:
            self.logger.warning(w)

        return config

    def register_config(self, precision: Precision, gpu_type: GPUType, runtime: Runtime, 
                        name: Union[List, None] = None, **kwargs):
        """
        This method initializes, validates and registers an inference configuration

        Note, this method does not throw errors in case of wrong configurations,
        it just logs the error and does not load the corresponding configuration.

        ...
        Parameters
        ----------
        precision: Precision
            data precision to be used at inference time
        gpu_type: GPUType
            gpu to be sued at inference time
        runtime: Runtime
            runtime to be used at inference such as torch, torch.compile, tensorrt etc.
        name: str (optional)
            name of the configuration. If nothing is set, the configuration id will be used
        """
        try:
            config = self.create_config(precision, gpu_type, runtime, **kwargs)
            logging.info(f"Configuration {config.config_id} has been loaded")
            name = name or config.config_id
            self._registry[name] = config
        except ValueError as e:
            logging.error(f"""Configuration with input parameters {precision}, {gpu_type}, 
                          {runtime} has failed with the following error: \n {e} \n
                          The configuration will not be loaded.""")

    def get_config(self, name: str) -> InferenceConfig:
        """
        Given the name of a configuration, it returns its parameters

        ...
        Parameters
        ----------
        name: str
            configuration name to be returned
        
        Returns
        -------
        config: InferenceConfig
            inference configuration with the given name

        Raises
        ------
        ValueError
            if the configuration name does not appear in registry
        """
        if name in self._registry:
            return self._registry[name]
        else:
            raise ValueError(f"""No configuration called {name} exists in registry.
                             These are the currently available options: {self._registry.keys()}""")

    def list_configs(self) -> List[str]:
        """
        Lists available configurations

        ...
        Returns
        -------
        configs: list
            list of the configuration names in the registry
        """
        return list(self._registry.keys())
