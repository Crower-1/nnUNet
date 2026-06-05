from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.network_architecture.plainconv_unet_head import PlainConvUNetHead
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class MultiHeadPlanner(ExperimentPlanner):
    """Planner that keeps the multi-head UNet while using the default automatic planning logic."""

    @staticmethod
    def _derive_class_names(dataset_json: dict) -> list[str]:
        label_manager = LabelManager(dataset_json['labels'], dataset_json.get('regions_class_order'))
        if label_manager.has_regions:
            class_names = [k for k in dataset_json['labels'].keys() if k not in ('background', 'ignore')]
        else:
            class_names = [k for k in dataset_json['labels'].keys() if k != 'ignore']
        if len(class_names) != label_manager.num_segmentation_heads:
            raise RuntimeError('Number of class names does not match number of segmentation heads')
        return class_names

    def get_plans_for_configuration(self, *args, **kwargs) -> dict:
        plan = super().get_plans_for_configuration(*args, **kwargs)
        architecture = plan['architecture']
        architecture['network_class_name'] = PlainConvUNetHead.__module__ + '.' + PlainConvUNetHead.__name__
        architecture['arch_kwargs']['class_names'] = self._derive_class_names(self.dataset_json)
        return plan
