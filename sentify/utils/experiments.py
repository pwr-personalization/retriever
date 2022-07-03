import datetime
import json
from pathlib import Path
from typing import Any

import torch
import wandb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, AutoModel, RobertaConfig, RobertaTokenizer,
)

from sentify import SENTILARE_DIR, LOGS_PATH, CHECKPOINTS_PATH, METRICS_PATH
from sentify.callbacks.wandb import WatchModel, LogParamsFile, LogConfusionMatrix
from sentify.data.augmentations.user_identifier import AUGMENTATIONS
from sentify.data.augmentations.user_retriever import (
    SentiLAREUserTextEmbeddings, BIEncoderComputeSimilarity, CrossEncoderComputeSimilarity,
)
from sentify.datasets import DATASETS
from sentify.datasets.base import BaseDataModule
from sentify.modeling.retriever import Retriever
from sentify.modeling.senti_lare import RobertaModel
from sentify.training.trainer import TransformerSentimentTrainer, RetrieverTrainer
from sentify.utils.downloading import safe_from_pretrained
from sentify.utils.seed import set_seeds


def run_experiment(
    config: dict[str, Any],
    model_trainer: LightningModule,
    datamodule: BaseDataModule,
    experiment_name: str,
    experiment_tag: str,
):
    EXP_NAME = f'{experiment_name}_{experiment_tag}_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    wandb_logger = WandbLogger(
        entity='amc',
        project='nlp-sentify',
        name=EXP_NAME,
        save_dir=str(LOGS_PATH),
    )

    wandb_logger.log_hyperparams(config)

    checkpoint_dir = CHECKPOINTS_PATH.joinpath(EXP_NAME)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    trainer = Trainer(
        max_epochs=config['max_epochs'],
        logger=wandb_logger,
        callbacks=[
            WatchModel(log_graph=False),
            LogParamsFile(),
            LogConfusionMatrix(num_classes=datamodule.num_classes, log_modes=('test',)),
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor="val/f1_score",
                mode="max",
            ),
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(
                patience=config['patience'],
                monitor="val/f1_score",
                mode="max",
            ),
        ],
        gpus=config['gpus'] if torch.cuda.is_available() else None,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        log_every_n_steps=5,
        precision=16 if config['gpus'] and torch.cuda.is_available() else 32
    )

    wandb.require(experiment="service")
    trainer.fit(
        model=model_trainer,
        datamodule=datamodule,
    )

    metrics, *_ = trainer.test(dataloaders=datamodule)
    wandb_logger.log_metrics({k: v for k, v in metrics.items()})

    metrics_dir = METRICS_PATH.joinpath(f'{experiment_name}_{experiment_tag}')
    metrics_dir.mkdir(exist_ok=True, parents=True)
    with metrics_dir.joinpath(f'test_{config["model_random_state"]}.json').open('w') as file:
        json.dump(metrics, file, indent=2)

    wandb_logger.experiment.finish()


def create_datamodule(config):
    tokenizer = safe_from_pretrained(AutoTokenizer, config['model'])
    dataset_name = config['dataset']
    datamodule_class = DATASETS[dataset_name]
    datamodule = datamodule_class(
        tokenizer=tokenizer,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        seed=config['data_random_state'],
        max_length=config['max_length'],
    )
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def create_user_identifier_datamodule(config):
    datamodule = create_datamodule(config)
    tokenizer = safe_from_pretrained(AutoTokenizer, config['model'])
    augmentation = AUGMENTATIONS[config['augmentations']](tokenizer, config['prefix_length'])
    datamodule.augmentations = augmentation
    return datamodule


def create_retriever_datamodule(
    config,
):
    datamodule = create_datamodule(config)
    device = f"cuda:{config['gpus'][0]}" if config['gpus'] and torch.cuda.is_available() else 'cpu'

    encoder_name = config['encoder_name']
    if encoder_name.startswith('sentence-transformers'):
        augmentation = BIEncoderComputeSimilarity(
            datamodule=datamodule,
            encoder=SentenceTransformer(
                model_name_or_path=encoder_name,
                device=device,
            ),
            name=encoder_name,
        )
    elif encoder_name.startswith('cross-encoder'):
        augmentation = CrossEncoderComputeSimilarity(
            datamodule=datamodule,
            encoder=CrossEncoder(
                model_name=encoder_name,
                device=device,
            ),
            name=encoder_name,
        )
    else:
        raise ValueError(encoder_name)

    datamodule.augmentations = augmentation
    return datamodule


def create_retriever_sentilare_datamodule(
    config,
    checkpoint_path: Path = SENTILARE_DIR.joinpath('sentiLARE_pretrained'),
):
    checkpoint_path = str(checkpoint_path)
    datamodule = create_datamodule(config)
    # SentiLARE's input embeddings include POS embedding, word-level sentiment polarity embedding,
    # and sentence-level sentiment polarity embedding (which is set to be unknown during fine-tuning).
    is_pos_embedding = True
    is_senti_embedding = True
    is_polarity_embedding = True

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
    tokenizer = tokenizer_class.from_pretrained(checkpoint_path)
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        from_tf=bool('.ckpt' in checkpoint_path),
        config=config_class.from_pretrained(checkpoint_path),
        pos_tag_embedding=is_pos_embedding,
        senti_embedding=is_senti_embedding,
        polarity_embedding=is_polarity_embedding,
    )
    device = f"cuda:{config['gpus'][0]}" if config['gpus'] and torch.cuda.is_available() else 'cpu'
    model.to(device)

    augmentation = SentiLAREUserTextEmbeddings(
        datamodule=datamodule,
        tokenizer=tokenizer,
        model=model,
        batch_size=config['batch_size'],
        max_seq_length=config['max_length'],
        mean_center_embeddings=config['mean_center_embeddings'],
        device=device,
    )
    datamodule.augmentations = augmentation
    return datamodule


def create_baseline_model_trainer(config, num_labels):
    set_seeds(config['model_random_state'])

    model = safe_from_pretrained(
        AutoModelForSequenceClassification,
        config['model'],
        num_labels=num_labels,
    )
    if config['gradient_checkpointing']:
        model.gradient_checkpointing_enable()

    if config['freeze_backbone']:
        _freeze(model)

    return TransformerSentimentTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        num_classes=num_labels,
    )


def create_retriever_model_trainer(config, num_labels):
    set_seeds(config['model_random_state'])

    backbone = safe_from_pretrained(
        AutoModel,
        config['model'],
        add_pooling_layer=False,
    )
    model = Retriever(
        num_labels=num_labels,
        backbone=backbone,
        normalize_weights=config['normalize_weights'],
        feature_normalization=config['feature_normalization'],
        top_k=config['top_k'],
    )

    if config['gradient_checkpointing']:
        model.backbone.gradient_checkpointing_enable()

    if config['freeze_backbone']:
        _freeze(model.backbone)

    return RetrieverTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        num_classes=num_labels,
    )


def _freeze(model):
    model.base_model.apply(lambda param: param.requires_grad_(False))
