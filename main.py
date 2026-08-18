import pytorch_lightning as pl
from pytorch_lightning import cli as pl_cli
import sys
import yaml

class CustomLightningCLI(pl_cli.LightningCLI):
    def before_fit(self):
        config_filename = str(self.config.fit.config[0])
        self.trainer.logger.experiment.log_artifact(self.trainer.logger.run_id, config_filename)

    def after_fit(self):
        pass
       

def main():
    cli = CustomLightningCLI(save_config_callback=None, auto_configure_optimizers=False, parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == '__main__':
    main()