from model import RandomModel
import lightning as L
from lightning.pytorch.loggers import WandbLogger

def main():
    logger = WandbLogger(project='NLP')
    model = RandomModel()

    trainer = L.Trainer(max_epochs=100, logger=False)
    trainer.fit(model)


if __name__ == "__main__":
    main()