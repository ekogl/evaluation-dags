import time
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.decorators import task, task_group
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta

from arbo.airflow.optimizer import ArboOptimizer
from arbo.utils.storage import MinioClient
from arbo.utils.monitoring import PrometheusClient
from arbo.utils.logger import get_logger

logger = get_logger("arbo.iisas_image_inference")

default_args = {
    "owner": "user",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

MINIO_ENDPOINT = "minio.stefan-dev.svc.cluster.local:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "image-classification-data"

NAMESPACE = "default"

minio_env_dict = {
    "MINIO_ENDPOINT": MINIO_ENDPOINT,
    "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
    "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
    "MINIO_SECURE": "false"
}

# NOTE: only as fallback, should not be used
NUM_OF_PICTURES = 8

with DAG(
        dag_id="image_classification_dag_inference_arbo",
        default_args=default_args,
        description="IISAS Image Classification Inference Pipeline",
        schedule=None,
        catchup=False,
        tags=["iisas", "arbo", "minio"],
        max_active_tasks=20,
) as dag:
    # setup task
    @task
    def prepare_pipeline_configs():
        optimizer = ArboOptimizer(namespace=NAMESPACE, is_local=False)

        cluster_load = PrometheusClient(NAMESPACE).get_cluster_load()

        logger.info(f"Cluster Load set to {cluster_load}")

        input_quantity = MinioClient.get_directory_size(
            endpoint_url=f"http://{MINIO_ENDPOINT}",
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket_name=MINIO_BUCKET,
            prefix="inference/input"
        )

        if not input_quantity:
            # TODO: figure out a better way to handle this case
            logger.info("Falling back to default (= NUM_OF_PICTURES * 1024 * 1024)")
            input_quantity = NUM_OF_PICTURES * 1024 * 1024

        configs = optimizer.get_task_configs("iisas_image_inference", input_quantity=input_quantity,
                                             cluster_load=cluster_load)
        s_opt = len(configs)

        calculated_gamma = configs[0]["gamma"]
        predicted_amdahl = configs[0]["amdahl_time"]
        predicted_residual = configs[0]["residual_prediction"]
        predicted_std = configs[0]["predicted_std"]

        logger.info(f"Configuration received: s={s_opt}, gamma={calculated_gamma}, predicted time: {predicted_amdahl + predicted_residual:2f}")

        configurations = []
        for i in range(s_opt):
            config = {
                "chunk_id": str(i),
                "offset_args": [
                    "--input_image_path", "inference/input",
                    "--output_image_path", f"inference/offsetted/{i}",
                    "--dx", "0", "--dy", "0",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", str(i),
                    "--num_tasks", str(s_opt),
                ],
                "crop_args": [
                    "--input_image_path", f"inference/offsetted/{i}",
                    "--output_image_path", f"inference/cropped/{i}",
                    "--left", "20", "--top", "20",
                    "--right", "330", "--bottom", "330",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
                ],
                "enhance_brightness_args": [
                    "--input_image_path", f"inference/cropped/{i}",
                    "--output_image_path", f"inference/enhanced_brightness/{i}",
                    "--factor", str(1.2),
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0",
                    "--num_tasks", "1",
                ],
                "enhance_contrast_args": [
                    "--input_image_path", f"inference/enhanced_brightness/{i}",
                    "--output_image_path", f"inference/enhanced_contrast/{i}",
                    "--factor", str(1.2),
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0",
                    "--num_tasks", "1",
                ],
                "rotate_args": [
                    "--input_image_path", f"inference/enhanced_contrast/{i}",
                    "--output_image_path", f"inference/rotated/{i}",
                    "--rotation", " ".join(["0", "90", "180", "270"]),
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0",
                    "--num_tasks", "1",
                ],
                "grayscale_args": [
                    "--input_image_path", f"inference/rotated/{i}",
                    "--output_image_path", f"inference/grayscaled",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0",
                    "--num_tasks", "1",
                ]
            }
            configurations.append(config)

        return {
            "configurations": configurations,
            "metadata": {
                "start_time": time.time(),
                "s": s_opt,
                "gamma": calculated_gamma,
                "cluster_load": cluster_load,
                "amdahl_time": predicted_amdahl,
                "pred_residual": predicted_residual,
                "pred_std": predicted_std
            }
        }


    @task
    def extract_configs(data: dict):
        return data["configurations"]


    @task
    def extract_metadata(data: dict):
        return data["metadata"]


    @task_group(group_id="preprocessing_pipeline")
    def image_pipeline_group(offset_args, crop_args, enhance_brightness_args, enhance_contrast_args, rotate_args, grayscale_args, chunk_id):

        offset = KubernetesPodOperator(
            task_id="offset",
            name="offset-task",
            namespace=NAMESPACE,
            image="kogsi/image_classification:offset",
            arguments=offset_args,
            env_vars=minio_env_dict,
            get_logs=True,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            node_selector={"kubernetes.io/worker": "worker"},
        )

        crop = KubernetesPodOperator(
            task_id="crop",
            name="crop-task",
            namespace=NAMESPACE,
            image="kogsi/image_classification:crop",
            arguments=crop_args,
            env_vars=minio_env_dict,
            get_logs=True,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            node_selector={"kubernetes.io/worker": "worker"},
        )

        enhance_brightness = KubernetesPodOperator(
            task_id="enhance_brightness",
            name="enhance_brightness-task",
            namespace=NAMESPACE,
            image="kogsi/image_classification:enhance-brightness",
            arguments=enhance_brightness_args,
            env_vars=minio_env_dict,
            get_logs=True,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            node_selector={"kubernetes.io/worker": "worker"},
        )

        enhance_contrast = KubernetesPodOperator(
            task_id="enhance_contrast",
            name="enhance_contrast-task",
            namespace=NAMESPACE,
            image="kogsi/image_classification:enhance-contrast",
            arguments=enhance_contrast_args,
            env_vars=minio_env_dict,
            get_logs=True,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            node_selector={"kubernetes.io/worker": "worker"},
        )

        rotate = KubernetesPodOperator(
            task_id="rotate",
            name="rotate-task",
            namespace=NAMESPACE,
            image="kogsi/image_classification:rotate",
            arguments=rotate_args,
            env_vars=minio_env_dict,
            get_logs=True,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            node_selector={"kubernetes.io/worker": "worker"},
        )

        grayscale = KubernetesPodOperator(
            task_id="grayscale",
            name="grayscale-task",
            namespace=NAMESPACE,
            image="kogsi/image_classification:to-grayscale",
            arguments=grayscale_args,
            env_vars=minio_env_dict,
            get_logs=True,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            node_selector={"kubernetes.io/worker": "worker"},
        )

        offset >> crop >> enhance_brightness >> enhance_contrast >> rotate >> grayscale

    classification_inference = KubernetesPodOperator(
        task_id="classification_inference_task",
        name="classification-inference-task",
        namespace=NAMESPACE,
        image="kogsi/image_classification:classification-inference-tf2",
        arguments=[
            "--saved_model_path", "models/",
            "--inference_data_path", "inference/grayscaled",
            "--output_result_path", "inference/results/inference_results.json",
            "--bucket_name", MINIO_BUCKET,
            "--workers", "4",
        ],
        env_vars=minio_env_dict,
        get_logs=True,
        is_delete_operator_pod=True,
        image_pull_policy="IfNotPresent",
        startup_timeout_seconds=600,  # increase time for startup (large image)
        node_selector={"kubernetes.io/worker": "worker"},
    )


    @task(trigger_rule=TriggerRule.ALL_SUCCESS)
    def report_feedback(metadata: dict, **context):
        optimizer = ArboOptimizer(namespace=NAMESPACE, is_local=False)

        dag_id = context["dag"].dag_id
        run_id = context["run_id"]

        local_start = context["dag_run"].start_date.timestamp()
        fallback_dur = time.time() - local_start

        optimizer.report_success(
            task_name="iisas_image_inference",
            s=metadata["s"],
            gamma=metadata["gamma"],
            cluster_load=metadata["cluster_load"],
            predicted_amdahl=metadata["amdahl_time"],
            predicted_residual=metadata["pred_residual"],
            predicted_std=metadata["pred_std"],
            dag_id=dag_id,
            run_id=run_id,
            target_id="preprocessing_pipeline",
            fallback_duration=fallback_dur,
            is_group=True
        )


    pipeline_configs = prepare_pipeline_configs()
    pod_config_list = extract_configs(pipeline_configs)
    pipeline_metadata = extract_metadata(pipeline_configs)

    pipeline_instances = image_pipeline_group.expand_kwargs(pod_config_list)

    # pipeline_instances >> classification_inference
    pipeline_instances >> classification_inference
    pipeline_instances >> report_feedback(pipeline_metadata)