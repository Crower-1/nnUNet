from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner


class MultiHeadPlanner(ExperimentPlanner):
    """Planner that keeps the multi-head UNet while using the default automatic planning logic."""
