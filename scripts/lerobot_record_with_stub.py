"""Run lerobot-record with no-op stubs for processor steps that exist
in the trained checkpoint's saved pipeline JSONs but not in upstream
lerobot==0.4.4's ProcessorStepRegistry. Each missing step is configured
`enabled=False` in the saved JSON, so pass-through is correct."""

from dataclasses import dataclass, field

from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("delta_actions_processor")
@dataclass
class DeltaActionsProcessorStub(ProcessorStep):
    enabled: bool = False
    exclude_joints: list = field(default_factory=list)
    action_names: list = field(default_factory=list)

    def __call__(self, transition):
        if self.enabled:
            raise NotImplementedError(
                "delta_actions_processor stub only supports enabled=False"
            )
        return transition

    def transform_features(self, features):
        return features

    def get_config(self):
        return {
            "enabled": self.enabled,
            "exclude_joints": list(self.exclude_joints),
            "action_names": list(self.action_names),
        }


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStub(ProcessorStep):
    enabled: bool = False

    def __call__(self, transition):
        if self.enabled:
            raise NotImplementedError(
                "absolute_actions_processor stub only supports enabled=False"
            )
        return transition

    def transform_features(self, features):
        return features

    def get_config(self):
        return {"enabled": self.enabled}


if __name__ == "__main__":
    from lerobot.scripts.lerobot_record import main
    main()
