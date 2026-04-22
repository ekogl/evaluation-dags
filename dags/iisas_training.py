from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from airflow.models import Variable

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

minio_env_dict = {
    "MINIO_ENDPOINT": MINIO_ENDPOINT,
    "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
    "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
    "MINIO_SECURE": "false"
}

with DAG(
    dag_id="image_classification_dag_inference",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["iisas", "minio"],
    max_active_tasks=42,
) as dag:

    NUM_PARALLEL_TASKS = int(Variable.get("image_classification_pod_count", default_var=1))

    # -------------------------------------------------------------------------
    # Group: preprocessing_pipeline
    # -------------------------------------------------------------------------
    with TaskGroup(group_id="preprocessing_pipeline") as preprocessing_group:
        for i in range(NUM_PARALLEL_TASKS):

            offset = KubernetesPodOperator(
                task_id=f"offset_{i}",
                name=f"offset-task-{i}",
                namespace=NAMESPACE,
                image="kogsi/image_classification:offset",
                arguments=[
                    "--input_image_path", "inference/input",
                    "--output_image_path", f"inference/offsetted/{i}",
                    "--dx", "0", "--dy", "0",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", str(i),
                    "--num_tasks", str(NUM_PARALLEL_TASKS),
                ],
                env_vars=minio_env_dict,
                get_logs=True,
                is_delete_operator_pod=True,
                image_pull_policy="IfNotPresent",
                node_selector={"kubernetes.io/worker": "worker"},
            )

            crop = KubernetesPodOperator(
                task_id=f"crop_{i}",
                name=f"crop-task-{i}",
                namespace=NAMESPACE,
                image="kogsi/image_classification:crop",
                arguments=[
                    "--input_image_path", f"inference/offsetted/{i}",
                    "--output_image_path", f"inference/cropped/{i}",
                    "--left", "20", "--top", "20",
                    "--right", "330", "--bottom", "330",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
                ],
                env_vars=minio_env_dict,
                get_logs=True,
                is_delete_operator_pod=True,
                image_pull_policy="IfNotPresent",
                node_selector={"kubernetes.io/worker": "worker"},
            )

            enhance_brightness = KubernetesPodOperator(
                task_id=f"enhance_brightness_{i}",
                name=f"enhance-brightness-task-{i}",
                namespace=NAMESPACE,
                image="kogsi/image_classification:enhance-brightness",
                arguments=[
                    "--input_image_path", f"inference/cropped/{i}",
                    "--output_image_path", f"inference/enhanced_brightness/{i}",
                    "--factor", str(1.2),
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
                ],
                env_vars=minio_env_dict,
                get_logs=True,
                is_delete_operator_pod=True,
                image_pull_policy="IfNotPresent",
                node_selector={"kubernetes.io/worker": "worker"},
            )

            enhance_contrast = KubernetesPodOperator(
                task_id=f"enhance_contrast_{i}",
                name=f"enhance-contrast-task-{i}",
                namespace=NAMESPACE,
                image="kogsi/image_classification:enhance-contrast",
                arguments=[
                    "--input_image_path", f"inference/enhanced_brightness/{i}",
                    "--output_image_path", f"inference/enhanced_contrast/{i}",
                    "--factor", str(1.2),
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
                ],
                env_vars=minio_env_dict,
                get_logs=True,
                is_delete_operator_pod=True,
                image_pull_policy="IfNotPresent",
                node_selector={"kubernetes.io/worker": "worker"},
            )

            rotate = KubernetesPodOperator(
                task_id=f"rotate_{i}",
                name=f"rotate-task-{i}",
                namespace=NAMESPACE,
                image="kogsi/image_classification:rotate",
                arguments=[
                    "--input_image_path", f"inference/enhanced_contrast/{i}",
                    "--output_image_path", f"inference/rotated/{i}",
                    "--rotation", "0",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
                ],
                env_vars=minio_env_dict,
                get_logs=True,
                is_delete_operator_pod=True,
                image_pull_policy="IfNotPresent",
                node_selector={"kubernetes.io/worker": "worker"},
            )

            grayscale = KubernetesPodOperator(
                task_id=f"grayscale_{i}",
                name=f"to-grayscale-task-{i}",
                namespace=NAMESPACE,
                image="kogsi/image_classification:to-grayscale",
                arguments=[
                    "--input_image_path", f"inference/rotated/{i}",
                    "--output_image_path", "inference/grayscaled",
                    "--bucket_name", MINIO_BUCKET,
                    "--chunk_id", "0", "--num_tasks", "1",
                ],
                env_vars=minio_env_dict,
                get_logs=True,
                is_delete_operator_pod=True,
                image_pull_policy="IfNotPresent",
                node_selector={"kubernetes.io/worker": "worker"},
            )

            offset >> crop >> enhance_brightness >> enhance_contrast >> rotate >> grayscale

    # -------------------------------------------------------------------------
    # classification_inference (downstream of full group)
    # -------------------------------------------------------------------------
    classification_inference_task = KubernetesPodOperator(
        task_id="classification_inference_task_training",
        name="classification-inference-task-training",
        namespace=NAMESPACE,
        image="kogsi/image_classification:classification-train-tf2",
        arguments=[
            "--train_data_path", "training/grayscaled",
            "--output_artifact_path", "models/",
            "--bucket_name", MINIO_BUCKET,
            "--validation_split", "0.2",
            # "--validation_data_path", "training/validation",
            "--epochs", "5",
            "--batch_size", "32",
            "--early_stop_patience", "5",
            "--dropout_rate", "0.2",
            "--image_size", "256 256",
            "--num_layers", "3",
            "--filters_per_layer", "64 64 64",
            "--kernel_sizes", "3 3 3",
            "--workers", "4",
        ],
        env_vars=minio_env_dict,
        get_logs=True,
        is_delete_operator_pod=True,
        image_pull_policy="Always",
        startup_timeout_seconds=600,  # increase time for startup (large image)
        node_selector={"kubernetes.io/worker": "worker"},
    )

    preprocessing_group >> classification_inference_task
    
    

    # )