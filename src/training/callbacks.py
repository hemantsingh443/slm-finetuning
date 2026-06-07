import wandb

class WandBLogger:
    """Helper class for logging experiments to Weights & Biases."""

    def __init__(self, project: str = "slm-finetuning", config: dict = None, name: str = None, **kwargs):
        self.project = project
        self.config = config
        self.name = name
        self.kwargs = kwargs

    def initialize(self):
        return wandb.init(
            project=self.project,
            config=self.config,
            name=self.name,
            **self.kwargs
        )

    def log(self, metrics: dict, step: int = None):
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()


def initialize_wandb(project: str = "slm-finetuning", config: dict = None, name: str = None, **kwargs):
    """Initializes a Weights & Biases run with options."""
    return wandb.init(
        project=project,
        config=config,
        name=name,
        **kwargs
    )
