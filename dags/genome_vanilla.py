from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.task_group import TaskGroup
from kubernetes.client import models as k8s
from datetime import datetime, timedelta
from airflow.models import Variable

default_args = {
    "owner": 'user',
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

TOTAL_ITEMS = 80000
MINIO_ENDPOINT = "minio.default.svc.cluster.local:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
CHROM_NR = "22"
MINIO_BUCKET = "genome-data"
KEY_INPUT_INDIVIDUAL = f"ALL.chr22.{TOTAL_ITEMS}.vcf.gz"
KEY_INPUT_SIFTING = "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5.20130502.sites.annotation.vcf.gz"

NAMESPACE = "default"
FREQ_TOTAL_PLOTS = 1000

minio_env_vars = [
    k8s.V1EnvVar(name="MINIO_ENDPOINT", value=MINIO_ENDPOINT),
    k8s.V1EnvVar(name="MINIO_ACCESS_KEY", value=MINIO_ACCESS_KEY),
    k8s.V1EnvVar(name="MINIO_SECRET_KEY", value=MINIO_SECRET_KEY),
    k8s.V1EnvVar(name="MINIO_SECURE", value="false"),
]

with DAG(
        dag_id='genome_vanilla',
        default_args=default_args,
        description='Genome processing pipeline using KubernetesPodOperator',
        schedule=None,
        catchup=False,
        tags=['genome', 'kubernetes', 'minio'],
        max_active_tasks=42,
) as dag:

    INDIVIDUAL_WORKERS = int(Variable.get("genome_individual_pod_count", default_var=1))
    FREQUENCY_WORKERS = int(Variable.get("genome_freq_pod_count", default_var=1))

    populations = ["EUR", "AFR", "EAS", "ALL", "GBR", "SAS", "AMR"]

    # -------------------------------------------------------------------------
    # Group: individual_tasks
    # -------------------------------------------------------------------------
    with TaskGroup(group_id="individual_tasks") as individual_group:
        individual_tasks = []
        for x in range(INDIVIDUAL_WORKERS):
            counter = x * TOTAL_ITEMS + 1
            stop = (x + 1) * TOTAL_ITEMS + 1

            task = KubernetesPodOperator(
                task_id=f"individual_{x}",
                name=f"individual-{x}",
                namespace=NAMESPACE,
                image="kogsi/genome_dag:individual",
                cmds=["python3", "individual.py"],
                arguments=[
                    "--key_input", KEY_INPUT_INDIVIDUAL,
                    "--counter", str(counter),
                    "--stop", str(stop),
                    "--chromNr", CHROM_NR,
                    "--bucket_name", MINIO_BUCKET
                ],
                env_vars=minio_env_vars,
                get_logs=True,
                is_delete_operator_pod=True,
                image_pull_policy="IfNotPresent",
                execution_timeout=timedelta(hours=1),
                node_selector={"kubernetes.io/worker": "worker"},
            )
            individual_tasks.append(task)

    # -------------------------------------------------------------------------
    # Sifting (standalone — not grouped, no parallelism to measure)
    # -------------------------------------------------------------------------
    sifting_task = KubernetesPodOperator(
        task_id="sifting",
        name="sifting",
        namespace=NAMESPACE,
        image="kogsi/genome_dag:sifting",
        cmds=["python3", "sifting.py"],
        arguments=[
            "--key_datafile", KEY_INPUT_SIFTING,
            "--chromNr", CHROM_NR,
            "--bucket_name", MINIO_BUCKET
        ],
        env_vars=minio_env_vars,
        get_logs=True,
        is_delete_operator_pod=True,
        image_pull_policy="IfNotPresent",
        execution_timeout=timedelta(hours=1),
        node_selector={"kubernetes.io/worker": "worker"},
    )

    # -------------------------------------------------------------------------
    # Group: individuals_merge (single task, but keeps collector prefix consistent)
    # -------------------------------------------------------------------------
    individuals_merge_task = KubernetesPodOperator(
        task_id="individuals_merge",
        name="individuals-merge",
        namespace=NAMESPACE,
        image="kogsi/genome_dag:individuals-merge",
        cmds=["python3", "individuals-merge.py"],
        arguments=[
            "--chromNr", CHROM_NR,
            "--keys", ','.join([
                f'chr22n-{x * TOTAL_ITEMS + 1}-{(x + 1) * TOTAL_ITEMS + 1}.tar.gz'
                for x in range(INDIVIDUAL_WORKERS)
            ]),
            "--bucket_name", MINIO_BUCKET
        ],
        env_vars=minio_env_vars,
        get_logs=True,
        is_delete_operator_pod=True,
        image_pull_policy="IfNotPresent",
        execution_timeout=timedelta(hours=1),
        node_selector={"kubernetes.io/worker": "worker"},
    )

    # -------------------------------------------------------------------------
    # Groups: freq_<POP> — one group per population
    # -------------------------------------------------------------------------
    mutations_overlap_tasks = []

    for pop in populations:
        with TaskGroup(group_id=f"freq_{pop}") as freq_group:
            if FREQUENCY_WORKERS > 1:
                freq_chunk_size = FREQ_TOTAL_PLOTS // FREQUENCY_WORKERS

                freq_workers = []
                for i in range(FREQUENCY_WORKERS):
                    start_idx = i * freq_chunk_size
                    end_idx = (i + 1) * freq_chunk_size if i < FREQUENCY_WORKERS - 1 else FREQ_TOTAL_PLOTS

                    freq_calc_plot = KubernetesPodOperator(
                        task_id=f"frequency_calc_plot_{i}",
                        name=f"frequency-calc-plot-{pop.lower()}-{i}",
                        namespace=NAMESPACE,
                        image="kogsi/genome_dag:frequency_par2",
                        cmds=["python3", "frequency_par2.py"],
                        arguments=[
                            "--mode", "calc_plot",
                            "--chromNr", CHROM_NR,
                            "--POP", pop,
                            "--bucket_name", MINIO_BUCKET,
                            "--start", str(start_idx),
                            "--end", str(end_idx),
                            "--chunk_id", str(i),
                        ],
                        env_vars=minio_env_vars,
                        get_logs=True,
                        is_delete_operator_pod=True,
                        image_pull_policy="IfNotPresent",
                        execution_timeout=timedelta(hours=1),
                        node_selector={"kubernetes.io/worker": "worker"},
                    )
                    freq_workers.append(freq_calc_plot)

                freq_merge = KubernetesPodOperator(
                    task_id="frequency_merge",
                    name=f"frequency-merge-{pop.lower()}",
                    namespace=NAMESPACE,
                    image="kogsi/genome_dag:frequency_par2",
                    cmds=["python3", "frequency_par2.py"],
                    arguments=[
                        "--mode", "merge",
                        "--chromNr", CHROM_NR,
                        "--POP", pop,
                        "--bucket_name", MINIO_BUCKET,
                        "--chunks", str(FREQUENCY_WORKERS),
                    ],
                    env_vars=minio_env_vars,
                    get_logs=True,
                    is_delete_operator_pod=True,
                    image_pull_policy="IfNotPresent",
                    execution_timeout=timedelta(hours=1),
                    node_selector={"kubernetes.io/worker": "worker"},
                )

                freq_workers >> freq_merge

            else:
                KubernetesPodOperator(
                    task_id="frequency",
                    name=f"frequency-{pop.lower()}",
                    namespace=NAMESPACE,
                    image="kogsi/genome_dag:frequency",
                    cmds=["python3", "frequency.py"],
                    arguments=[
                        "--chromNr", CHROM_NR,
                        "--POP", pop,
                        "--bucket_name", MINIO_BUCKET
                    ],
                    env_vars=minio_env_vars,
                    get_logs=True,
                    is_delete_operator_pod=False,
                    image_pull_policy="IfNotPresent",
                    execution_timeout=timedelta(hours=1),
                    node_selector={"kubernetes.io/worker": "worker"},
                )

        # Wire upstream → freq group
        individuals_merge_task >> freq_group
        sifting_task >> freq_group

        # Mutations overlap per population (outside freq group)
        mutations_overlap = KubernetesPodOperator(
            task_id=f"mutations_overlap_{pop}",
            name=f"mutations-overlap-{pop.lower()}",
            namespace=NAMESPACE,
            image="kogsi/genome_dag:mutations-overlap",
            cmds=["python3", "mutations-overlap.py"],
            arguments=[
                "--chromNr", CHROM_NR,
                "--POP", pop,
                "--bucket_name", MINIO_BUCKET
            ],
            env_vars=minio_env_vars,
            get_logs=True,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            execution_timeout=timedelta(hours=1),
            node_selector={"kubernetes.io/worker": "worker"},
        )
        mutations_overlap_tasks.append(mutations_overlap)

    # -------------------------------------------------------------------------
    # Wire remaining dependencies
    # -------------------------------------------------------------------------
    individual_group >> individuals_merge_task
    individuals_merge_task >> mutations_overlap_tasks
    sifting_task >> mutations_overlap_tasks
