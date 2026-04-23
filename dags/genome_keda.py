from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.task_group import TaskGroup
from airflow.decorators import task
from kubernetes.client import models as k8s
from datetime import datetime, timedelta

from arbo.utils.storage import MinioClient

default_args = {
    "owner": 'user',
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

TOTAL_ITEMS = 15000
MINIO_ENDPOINT = "minio.default.svc.cluster.local:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
CHROM_NR = "22"
MINIO_BUCKET = "genome-data"
KEY_INPUT_INDIVIDUAL = f"ALL.chr22.{TOTAL_ITEMS}.vcf.gz"
KEY_INPUT_SIFTING = "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5.20130502.sites.annotation.vcf.gz"

NAMESPACE = "default"
FREQ_TOTAL_PLOTS = 1000

# Tuning knobs — adjust these to control scaling sensitivity
INDIVIDUAL_BYTES_PER_WORKER = 5 * 1024 * 1024   # 1 worker per 5 MB
FREQ_ITEMS_PER_WORKER       = 100                # 1 worker per 100 plots
MAX_INDIVIDUAL_WORKERS      = 15
MAX_FREQ_WORKERS            = 10

minio_env_vars = [
    k8s.V1EnvVar(name="MINIO_ENDPOINT",   value=MINIO_ENDPOINT),
    k8s.V1EnvVar(name="MINIO_ACCESS_KEY", value=MINIO_ACCESS_KEY),
    k8s.V1EnvVar(name="MINIO_SECRET_KEY", value=MINIO_SECRET_KEY),
    k8s.V1EnvVar(name="MINIO_SECURE",     value="false"),
]

with DAG(
        dag_id='genome_keda',
        default_args=default_args,
        description='Genome processing pipeline using KubernetesPodOperator',
        schedule=None,
        catchup=False,
        tags=['genome', 'kubernetes', 'minio'],
        max_active_tasks=42,
) as dag:

    populations = ["EUR", "AFR", "EAS", "ALL", "GBR", "SAS", "AMR"]

    # =========================================================================
    # PREPARE TASKS — run at runtime, return argument lists for .expand()
    # =========================================================================

    @task()
    def prepare_individual_tasks() -> dict:
        input_bytes = MinioClient.get_filesize(
            endpoint_url=f"http://{MINIO_ENDPOINT}",
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket_name=MINIO_BUCKET,
            file_key=f"input/{KEY_INPUT_INDIVIDUAL}",
        )
        if not input_bytes:
            raise ValueError(f"Input file not found in MinIO: input/{KEY_INPUT_INDIVIDUAL}")

        # KEDA-style: scale workers linearly with input size, capped at max
        num_workers = min(
            max(1, input_bytes // INDIVIDUAL_BYTES_PER_WORKER),
            MAX_INDIVIDUAL_WORKERS,
        )
        chunk_size = TOTAL_ITEMS // num_workers

        pod_args, merge_keys = [], []
        for i in range(num_workers):
            counter = i * chunk_size + 1
            stop    = TOTAL_ITEMS + 1 if i == num_workers - 1 else (i + 1) * chunk_size + 1
            pod_args.append([
                "--key_input", KEY_INPUT_INDIVIDUAL,
                "--counter",   str(counter),
                "--stop",      str(stop),
                "--chromNr",   CHROM_NR,
                "--bucket_name", MINIO_BUCKET,
            ])
            merge_keys.append(f"chr22n-{counter}-{stop}.tar.gz")

        print(f"[individual] input={input_bytes}B → {num_workers} workers, chunk={chunk_size}")
        return {
            "pod_args":   pod_args,
            "merge_keys": ",".join(merge_keys),
            "num_workers": num_workers,
        }

    @task()
    def prepare_frequency_tasks(pop: str) -> dict:

        num_workers = min(
            max(1, FREQ_TOTAL_PLOTS // FREQ_ITEMS_PER_WORKER),
            MAX_FREQ_WORKERS,
        )
        chunk_size = FREQ_TOTAL_PLOTS // num_workers

        worker_args = []
        for i in range(num_workers):
            start = i * chunk_size
            end   = (i + 1) * chunk_size if i < num_workers - 1 else FREQ_TOTAL_PLOTS
            worker_args.append([
                "--mode",        "calc_plot",
                "--chromNr",     CHROM_NR,
                "--POP",         pop,
                "--bucket_name", MINIO_BUCKET,
                "--start",       str(start),
                "--end",         str(end),
                "--chunk_id",    str(i),
            ])

        merger_args = [[
            "--mode",        "merge",
            "--chromNr",     CHROM_NR,
            "--POP",         pop,
            "--bucket_name", MINIO_BUCKET,
            "--chunks",      str(num_workers),
        ]]

        print(f"[freq/{pop}] scale_input={scale_input} → {num_workers} workers")
        return {
            "worker_args": worker_args,
            "merger_args": merger_args,
        }

    # Extractor helpers (needed because .expand() requires a plain list, not a dict)
    @task()
    def get_pod_args(d: dict):   return d["pod_args"]
    @task()
    def get_merge_keys(d: dict): return d["merge_keys"]
    @task()
    def get_worker_args(d: dict): return d["worker_args"]
    @task()
    def get_merger_args(d: dict): return d["merger_args"]

    # =========================================================================
    # GROUP: individual_tasks
    # =========================================================================
    with TaskGroup(group_id="individual_tasks") as individual_group:
        ind_plan = prepare_individual_tasks()

        individual_workers = KubernetesPodOperator.partial(
            task_id="worker",
            name="individual-worker",
            namespace=NAMESPACE,
            image="kogsi/genome_dag:individual",
            cmds=["python3", "individual.py"],
            env_vars=minio_env_vars,
            get_logs=True,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            execution_timeout=timedelta(hours=1),
            node_selector={"kubernetes.io/worker": "worker"},
        ).expand(arguments=get_pod_args(ind_plan))

        individuals_merge = KubernetesPodOperator(
            task_id="merge",
            name="individuals-merge",
            namespace=NAMESPACE,
            image="kogsi/genome_dag:individuals-merge",
            cmds=["python3", "individuals-merge.py"],
            arguments=[
                "--chromNr",     CHROM_NR,
                "--keys",        get_merge_keys(ind_plan),
                "--bucket_name", MINIO_BUCKET,
            ],
            env_vars=minio_env_vars,
            get_logs=True,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            execution_timeout=timedelta(hours=1),
            node_selector={"kubernetes.io/worker": "worker"},
        )

        individual_workers >> individuals_merge

    # =========================================================================
    # Sifting (standalone)
    # =========================================================================
    sifting_task = KubernetesPodOperator(
        task_id="sifting",
        name="sifting",
        namespace=NAMESPACE,
        image="kogsi/genome_dag:sifting",
        cmds=["python3", "sifting.py"],
        arguments=[
            "--key_datafile", KEY_INPUT_SIFTING,
            "--chromNr",      CHROM_NR,
            "--bucket_name",  MINIO_BUCKET,
        ],
        env_vars=minio_env_vars,
        get_logs=True,
        is_delete_operator_pod=True,
        image_pull_policy="IfNotPresent",
        execution_timeout=timedelta(hours=1),
        node_selector={"kubernetes.io/worker": "worker"},
    )

    # =========================================================================
    # GROUPS: freq_<POP>
    # =========================================================================
    mutations_overlap_tasks = []

    for pop in populations:
        with TaskGroup(group_id=f"freq_{pop}") as freq_group:
            freq_plan = prepare_frequency_tasks(pop)

            freq_workers = KubernetesPodOperator.partial(
                task_id="worker",
                name=f"freq-worker-{pop.lower()}",
                namespace=NAMESPACE,
                image="kogsi/genome_dag:frequency_par2",
                cmds=["python3", "frequency_par2.py"],
                env_vars=minio_env_vars,
                get_logs=True,
                is_delete_operator_pod=True,
                image_pull_policy="IfNotPresent",
                execution_timeout=timedelta(hours=1),
                node_selector={"kubernetes.io/worker": "worker"},
            ).expand(arguments=get_worker_args(freq_plan))

            freq_merge = KubernetesPodOperator.partial(
                task_id="merge",
                name=f"freq-merge-{pop.lower()}",
                namespace=NAMESPACE,
                image="kogsi/genome_dag:frequency_par2",
                cmds=["python3", "frequency_par2.py"],
                env_vars=minio_env_vars,
                get_logs=True,
                is_delete_operator_pod=True,
                image_pull_policy="IfNotPresent",
                execution_timeout=timedelta(hours=1),
                node_selector={"kubernetes.io/worker": "worker"},
            ).expand(arguments=get_merger_args(freq_plan))

            freq_workers >> freq_merge

        individual_group >> freq_group
        sifting_task    >> freq_group

        # Mutations overlap — one per population, outside the freq group
        mutations_overlap_tasks.append(KubernetesPodOperator(
            task_id=f"mutations_overlap_{pop}",
            name=f"mutations-overlap-{pop.lower()}",
            namespace=NAMESPACE,
            image="kogsi/genome_dag:mutations-overlap",
            cmds=["python3", "mutations-overlap.py"],
            arguments=[
                "--chromNr",     CHROM_NR,
                "--POP",         pop,
                "--bucket_name", MINIO_BUCKET,
            ],
            env_vars=minio_env_vars,
            get_logs=True,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            execution_timeout=timedelta(hours=1),
            node_selector={"kubernetes.io/worker": "worker"},
        ))

    # =========================================================================
    # Final wiring
    # =========================================================================
    individual_group >> mutations_overlap_tasks
    sifting_task     >> mutations_overlap_tasks