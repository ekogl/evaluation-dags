from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.task_group import TaskGroup
from airflow.decorators import task, task_group
from datetime import datetime, timedelta

from arbo.utils.storage import MinioClient

default_args = {
    "owner": 'user',
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

# Tuning knob: 1 worker per N bytes of input
BYTES_PER_WORKER = 10 * 1024 * 1024   # 1 worker per 10 MB
MAX_WORKERS      = 15  # TODO check if it makes sense
FALLBACK_WORKERS = 1

minio_env_dict = {
    "MINIO_ENDPOINT":   MINIO_ENDPOINT,
    "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
    "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
    "MINIO_SECURE":     "false",
}

with DAG(
    dag_id="iisas_keda",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["iisas", "minio"],
    max_active_tasks=42,
) as dag:

    @task()
    def prepare_pipeline_configs() -> list:
        input_bytes = MinioClient.get_directory_size(
            endpoint_url=f"http://{MINIO_ENDPOINT}",
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket_name=MINIO_BUCKET,
            prefix="inference/input",
        )

        if not input_bytes:
            print(f"MinIO directory size unavailable, falling back to {FALLBACK_WORKERS} worker(s)")
            num_workers = FALLBACK_WORKERS
        else:
            num_workers = min(max(1, input_bytes // BYTES_PER_WORKER), MAX_WORKERS)

        print(f"[image_classification] input={input_bytes}B → {num_workers} workers")

        # Build one config dict per chunk — passed directly to expand_kwargs()
        configs = []
        for i in range(num_workers):
            configs.append({
                "chunk_id": str(i),
                "offset_args": [
                    "--input_image_path", "inference/input",
                    "--output_image_path", f"inference/offsetted/{i}/",
                    "--dx", "0", "--dy", "0",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", str(i),
                    "--num_tasks", str(num_workers),
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
                    "--factor", "1.2",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
                ],
                "enhance_contrast_args": [
                    "--input_image_path", f"inference/enhanced_brightness/{i}/",
                    "--output_image_path", f"inference/enhanced_contrast/{i}/",
                    "--factor", "1.2",
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
                    "--output_image_path", "inference/grayscaled",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
                ],
            })

        return configs

    @task_group(group_id="preprocessing_pipeline")
    def image_pipeline_group(
        chunk_id,
        offset_args, crop_args,
        enhance_brightness_args, enhance_contrast_args,
        rotate_args, grayscale_args,
    ):
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
            name="enhance-brightness-task",
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
            name="enhance-contrast-task",
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

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------
    configs = prepare_pipeline_configs()
    pipeline_instances = image_pipeline_group.expand_kwargs(configs)

    classification_inference = KubernetesPodOperator(
        task_id="classification_inference",
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
        startup_timeout_seconds=600,
        node_selector={"kubernetes.io/worker": "worker"},
    )

    pipeline_instances >> classification_inference