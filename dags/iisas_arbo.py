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

MINIO_ENDPOINT = "minio.default.svc.cluster.local:9000"
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

NUM_OF_PICTURES = 8

with DAG(
        dag_id="iisas_arbo",
        default_args=default_args,
        description="IISAS Image Classification Inference Pipeline",
        schedule=None,
        catchup=False,
        tags=["iisas", "arbo", "minio"],
        max_active_tasks=20,
) as dag:

    # =================================
    # HELPER TASKS
    # =================================
    @task(trigger_rule=TriggerRule.ALL_SUCCESS)
    def report_feedback(metadata: dict, task_name: str, target_group_id: str, is_group: bool, **context):
        optimizer = ArboOptimizer(namespace=NAMESPACE, is_local=False)
        dag_id = context["dag"].dag_id
        run_id = context["run_id"]

        # FIX: Using metadata["start_time"] for accurate task group duration
        local_start = metadata["start_time"]
        fallback_dur = time.time() - local_start

        optimizer.report_success(
            task_name=task_name,
            s=metadata["s"],
            gamma=metadata["gamma"],
            cluster_load=metadata["cluster_load"],
            predicted_amdahl=metadata["amdahl_time"],
            predicted_residual=metadata["pred_residual"],
            predicted_std=metadata["pred_std"],
            dag_id=dag_id,
            run_id=run_id,
            target_id=target_group_id,
            fallback_duration=fallback_dur,
            is_group=is_group
        )

    @task
    def get_w_args(data: dict):
        return data["workers"]

    @task
    def get_m_args(data: dict):
        return data["merger"]

    @task
    def extract_configs(data: dict):
        return data["configurations"]

    @task
    def extract_metadata(data: dict):
        return data["metadata"]


    # =================================
    # PREPARATION TASKS
    # =================================
    @task()
    def prepare_preprocessing_configs():
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
            logger.info("Falling back to default (= NUM_OF_PICTURES * 1024 * 1024)")
            input_quantity = NUM_OF_PICTURES * 1024 * 1024

        configs = optimizer.get_task_configs("iisas_preprocessing", input_quantity=input_quantity,
                                             cluster_load=cluster_load)
        s_opt = len(configs)

        calculated_gamma = configs[0]["gamma"]
        predicted_amdahl = configs[0]["amdahl_time"]
        predicted_residual = configs[0]["residual_prediction"]
        predicted_std = configs[0]["predicted_std"]

        logger.info(f"Preprocessing Plan: s={s_opt}, gamma={calculated_gamma}, predicted time: {predicted_amdahl + predicted_residual:2f}")

        configurations = []
        for i in range(s_opt):
            config = {
                "chunk_id": str(i),
                "offset_args": [
                    "--input_image_path", "inference/input",
                    "--output_image_path", f"inference/offsetted/{i}/",
                    "--dx", "0", "--dy", "0",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", str(i),
                    "--num_tasks", str(s_opt),
                ],
                "crop_args": [
                    "--input_image_path", f"inference/offsetted/{i}/",
                    "--output_image_path", f"inference/cropped/{i}/",
                    "--left", "20", "--top", "20",
                    "--right", "330", "--bottom", "330",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
                ],
                "enhance_brightness_args": [
                    "--input_image_path", f"inference/cropped/{i}/",
                    "--output_image_path", f"inference/enhanced_brightness/{i}/",
                    "--factor", str(1.2),
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
                ],
                "enhance_contrast_args": [
                    "--input_image_path", f"inference/enhanced_brightness/{i}/",
                    "--output_image_path", f"inference/enhanced_contrast/{i}/",
                    "--factor", str(1.2),
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
                ],
                "rotate_args": [
                    "--input_image_path", f"inference/enhanced_contrast/{i}/",
                    "--output_image_path", f"inference/rotated/{i}/",
                    "--rotation", "0",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
                ],
                "grayscale_args": [
                    "--input_image_path", f"inference/rotated/{i}/",
                    "--output_image_path", f"inference/grayscaled",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
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
    def prepare_inference_configs():
        optimizer = ArboOptimizer(namespace=NAMESPACE, is_local=False)
        cluster_load = PrometheusClient(NAMESPACE).get_cluster_load()

        logger.info(f"Cluster Load set to {cluster_load}")

        input_quantity = MinioClient.get_directory_size(
            endpoint_url=f"http://{MINIO_ENDPOINT}",
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket_name=MINIO_BUCKET,
            prefix="inference/grayscaled"
        )

        if not input_quantity:
            logger.info("Falling back to default (= NUM_OF_PICTURES * 1024 * 1024)")
            input_quantity = NUM_OF_PICTURES * 1024 * 1024

        configs = optimizer.get_task_configs("iisas_inference", input_quantity=input_quantity,
                                             cluster_load=cluster_load)
        s_opt = len(configs)

        calculated_gamma = configs[0]["gamma"]
        predicted_amdahl = configs[0]["amdahl_time"]
        predicted_residual = configs[0]["residual_prediction"]
        predicted_std = configs[0]["predicted_std"]

        logger.info(f"Inference Plan: s={s_opt}, gamma={calculated_gamma}, predicted time: {predicted_amdahl + predicted_residual:2f}")

        worker_args = []
        for i in range(s_opt):
            worker_args.append([
                "--mode", "inference",
                "--saved_model_path", "models/",
                "--inference_data_path", "inference/grayscaled",
                "--output_result_path", f"inference/results/inference_results_{i}.json", 
                "--bucket_name", MINIO_BUCKET,
                "--workers", "4",
                "--batch_size", "32",
                "--chunk_id", str(i),
                "--num_tasks", str(s_opt)
            ])

        # Merge is a single task, so we wrap it in a list to use with get_m_args helper
        merger_args = [
            "--mode", "merge",
            "--input_results_prefix", "inference/results/", 
            "--output_result_path", "inference/final_merged/inference_results.json", 
            "--bucket_name", MINIO_BUCKET,
        ]

        return {
            "workers": worker_args,
            "merger": merger_args,
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


    # =================================
    # TASK GROUP DEFINITIONS
    # =================================
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

    @task_group(group_id="inference_pipeline")
    def run_inference_pipeline():
        inference_plan = prepare_inference_configs()

        workers = KubernetesPodOperator.partial(
            task_id="classification_inference",
            name="classification-inference-task",
            namespace=NAMESPACE,
            image="kogsi/image_classification:classification-inference-tf2",
            env_vars=minio_env_dict,
            get_logs=True,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            startup_timeout_seconds=600,
            node_selector={"kubernetes.io/worker": "worker"},
        ).expand(
            arguments=get_w_args(inference_plan)
        )

        # Merge results task (single execution)
        merge_results = KubernetesPodOperator(
            task_id="merge_results",
            name="merge-results-task",
            namespace=NAMESPACE,
            image="kogsi/image_classification:classification-inference-tf2", 
            arguments=[
                "--mode", "merge",
                "--input_results_prefix", "inference/results/", 
                "--output_result_path", "inference/final_merged/inference_results.json", 
                "--bucket_name", MINIO_BUCKET,
            ],
            env_vars=minio_env_dict,
            get_logs=True,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            node_selector={"kubernetes.io/worker": "worker"},
        )

        inference_feedback = report_feedback(
            metadata=extract_metadata(inference_plan),
            task_name="iisas_inference",
            target_group_id="inference_pipeline",
            is_group=True
        )

        workers >> merge_results >> inference_feedback


    # =================================
    # WIRING OF DAG
    # =================================
    
    # 1. Prepare and Run Preprocessing
    pipeline_configs = prepare_preprocessing_configs()
    pod_config_list = extract_configs(pipeline_configs)
    pipeline_metadata = extract_metadata(pipeline_configs)

    preprocessing_instances = image_pipeline_group.expand_kwargs(pod_config_list)
    
    preprocessing_feedback = report_feedback(
        metadata=pipeline_metadata,
        task_name="iisas_preprocessing",
        target_group_id="preprocessing_pipeline",
        is_group=True
    )

    preprocessing_instances >> preprocessing_feedback

    inference_group = run_inference_pipeline()

    preprocessing_feedback >> inference_group