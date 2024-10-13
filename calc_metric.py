from src.utils.io_utils import ROOT_PATH


@hydra.main(version_base=None, config_path="src/configs", config_name="calc_metric")
def main(config):
    gt_path = config.paths.ground_truth
    pred_path = config.paths.predictions

    text_encoder = instantiate(config.text_encoder)

    metrics = []
    for metric_config in config.metrics.get("inference"):
        metrics.append(instantiate(metric_config, text_encoder=text_encoder))


if __name__ == "main":
    main()
