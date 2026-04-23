import time
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.decorators import task, task_group
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from kubernetes.client import models as k8s
from datetime import datetime, timedelta

from arbo.airflow.optimizer import ArboOptimizer
from arbo.utils.storage import MinioClient
from arbo.utils.monitoring import PrometheusClient
from arbo.utils.logger import get_logger


logger = get_logger("arbo.genome_dag")

default_args = {
    "owner": "user",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

TOTAL_ITEMS = 15000
FREQ_TOTAL_PLOTS = 1000

MINIO_ENDPOINT = "minio.default.svc.cluster.local:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
CHROM_NR = "22"
MINIO_BUCKET = "genome-data"
KEY_INPUT_INDIVIDUAL = f"ALL.chr22.{TOTAL_ITEMS}.vcf.gz"
KEY_INPUT_SIFTING = "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5.20130502.sites.annotation.vcf.gz"

NAMESPACE = "default"

minio_env_vars = [
    k8s.V1EnvVar(name="MINIO_ENDPOINT", value=MINIO_ENDPOINT),
    k8s.V1EnvVar(name="MINIO_ACCESS_KEY", value=MINIO_ACCESS_KEY),
    k8s.V1EnvVar(name="MINIO_SECRET_KEY", value=MINIO_SECRET_KEY),
    k8s.V1EnvVar(name="MINIO_SECURE", value="false"),
]

with DAG(
        dag_id="genome_data_processing_arbo",
        default_args=default_args,
        description="Genome processing pipeline",
        schedule=None,
        catchup=False,
        tags=["genome", "arbo", 'minio'],
        max_active_tasks=20,
) as dag:

    # NOTE: currently decreases number of populations for cluster reasons
    populations = ["EUR", "AFR", "EAS", "ALL", "GBR", "SAS", "AMR"]
    # populations = ["AFR", "ALL"]


    # =================================
    # HELPER TASKS
    # =================================
    @task(trigger_rule=TriggerRule.ALL_SUCCESS)
    def report_feedback(metadata: dict, task_name: str, target_group_id: str, is_group: bool, **context):
        optimizer = ArboOptimizer(namespace=NAMESPACE, is_local=False)
        dag_id = context["dag"].dag_id
        run_id = context["run_id"]

        local_start = context["dag_run"].start_date.timestamp()
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
    def extract_pod_args(data: dict):
        return data["pod_arguments"]


    @task
    def extract_merge_keys(data: dict):
        return data["merge_keys_str"]


    # preparation tasks
    @task()
    def prepare_individual_tasks():
        optimizer = ArboOptimizer(namespace=NAMESPACE, is_local=False)

        cluster_load = PrometheusClient(NAMESPACE).get_cluster_load()

        logger.info(f"Cluster Load set to {cluster_load}")

        input_quantity = MinioClient.get_filesize(
            endpoint_url=f"http://{MINIO_ENDPOINT}",
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket_name=MINIO_BUCKET,
            file_key=f"input/{KEY_INPUT_INDIVIDUAL}"
        )

        if not input_quantity:
            logger.info("Falling back to default (= TOTAL_ITEMS)")
            input_quantity = TOTAL_ITEMS

        configs = optimizer.get_task_configs("genome_individual", input_quantity=input_quantity,
                                             cluster_load=cluster_load)
        s_opt = len(configs)

        calculated_gamma = configs[0]["gamma"]
        predicted_amdahl = configs[0]["amdahl_time"]
        predicted_residual = configs[0]["residual_prediction"]
        predicted_std = configs[0]["predicted_std"]

        chunk_size = TOTAL_ITEMS // s_opt

        # generate arguments for each pod
        pod_argument_list = []
        merge_keys = []

        for i in range(s_opt):
            counter = i * chunk_size + 1
            if i == s_opt - 1:
                stop = TOTAL_ITEMS + 1
            else:
                stop = (i + 1) * chunk_size + 1

            args = [
                "--key_input", KEY_INPUT_INDIVIDUAL,
                "--counter", str(counter),
                "--stop", str(stop),
                "--chromNr", CHROM_NR,
                "--bucket_name", MINIO_BUCKET
            ]
            pod_argument_list.append(args)

            # prepare filename key for downstream tasks
            file_key = f'chr22n-{counter}-{stop}.tar.gz'
            merge_keys.append(file_key)

        logger.info(f"PLAN: s={s_opt}, chunk_size={chunk_size}, predicted time: {predicted_amdahl + predicted_residual:2f}")

        return {
            "pod_arguments": pod_argument_list,
            "merge_keys_str": ",".join(merge_keys),
            "s": s_opt,
            "start_time": time.time(),
            "gamma": calculated_gamma,
            "cluster_load": cluster_load,
            "amdahl_time": predicted_amdahl,
            "pred_residual": predicted_residual,
            "pred_std": predicted_std
        }


    @task
    def prepare_frequency_tasks(pop: str):
        optimizer = ArboOptimizer(namespace=NAMESPACE, is_local=False)

        cluster_load = PrometheusClient(NAMESPACE).get_cluster_load()

        logger.info(f"Cluster Load set to {cluster_load}")

        pop_input_size = MinioClient.get_filesize(
            endpoint_url=f"http://{MINIO_ENDPOINT}",
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket_name=MINIO_BUCKET,
            file_key=f"input/{pop}"
        )

        if not pop_input_size:
            logger.info(f"Falling back to default for population {pop}")
            pop_input_size = TOTAL_ITEMS  # TODO: change this will break if once successful and once not

        configs = optimizer.get_task_configs(f"genome_frequency_{pop}", input_quantity=pop_input_size,
                                             cluster_load=cluster_load)
        s_opt = len(configs)

        calculated_gamma = configs[0]["gamma"]
        predicted_amdahl = configs[0]["amdahl_time"]
        predicted_residual = configs[0]["residual_prediction"]
        predicted_std = configs[0]["predicted_std"]

        chunk_size = FREQ_TOTAL_PLOTS // s_opt

        logger.info(
            f"Population {pop}: Size={pop_input_size}, Optimal num Workers={s_opt}, Gamma={calculated_gamma}, Chunk Size={chunk_size}"
        )

        worker_args = []
        chunk_size = FREQ_TOTAL_PLOTS // s_opt

        for i in range(s_opt):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < s_opt - 1 else FREQ_TOTAL_PLOTS

            worker_args.append([
                "--mode", "calc_plot",
                "--chromNr", CHROM_NR,
                "--POP", pop,
                "--bucket_name", MINIO_BUCKET,
                "--start", str(start),
                "--end", str(end),
                "--chunk_id", str(i)
            ])

        merger_args = [[
            "--mode", "merge",
            "--chromNr", CHROM_NR,
            "--POP", pop,
            "--bucket_name", MINIO_BUCKET,
            "--chunks", str(s_opt)
        ]]

        logger.info(f"Plan for {pop}: s={s_opt}, Size={pop_input_size}, predicted time: {predicted_amdahl + predicted_residual:2f}")

        return {
            "workers": worker_args,
            "merger": merger_args,
            "start_time": time.time(),
            "s": s_opt,
            "gamma": calculated_gamma,
            "cluster_load": cluster_load,
            "pop": pop,
            "amdahl_time": predicted_amdahl,
            "pred_residual": predicted_residual,
            "pred_std": predicted_std
        }


    @task
    def mutations_overlap_data(pops: list):
        data = []
        for pop in pops:
            data.append([
                "--chromNr", CHROM_NR,
                "--POP", pop,
                "--bucket_name", MINIO_BUCKET,
            ])
        return data


    # =================================
    # TASK GROUP DEFINITIONS
    # =================================
    @task_group(group_id="individual_tasks")
    def run_individual_tasks():
        ind_plan = prepare_individual_tasks()

        workers = KubernetesPodOperator.partial(
            task_id="workers",
            name="individual-worker",
            namespace=NAMESPACE,
            image="kogsi/genome_dag:individual",
            cmds=["python3", "individual.py"],
            env_vars=minio_env_vars,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            node_selector={"node-role.kubernetes.io/worker": "worker"},
        ).expand(
            arguments=extract_pod_args(ind_plan)
        )

        individual_merge = KubernetesPodOperator(
            task_id="merge",
            name="individuals_merge",
            namespace=NAMESPACE,
            image="kogsi/genome_dag:individuals-merge",
            cmds=["python3", "individuals-merge.py"],
            arguments=[
                "--chromNr", CHROM_NR,
                "--keys", extract_merge_keys(ind_plan),
                "--bucket_name", MINIO_BUCKET
            ],
            env_vars=minio_env_vars,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            node_selector={"node-role.kubernetes.io/worker": "worker"},
        )

        feedback = report_feedback(ind_plan, "genome_individual", "individual_tasks.workers", True)

        # HINT: currently merge and feedback run in parallel, meaning only individual time is accounted for
        workers >> individual_merge >> feedback
        # [individual_merge, feedback] << workers


    # helper to run frequency tasks
    def run_frequency_tasks(pop: str):
        plan_data = prepare_frequency_tasks(pop)

        workers = KubernetesPodOperator.partial(
            task_id="workers",
            name=f"freq-workers-{pop.lower()}",
            namespace=NAMESPACE,
            image="kogsi/genome_dag:frequency_par2",
            cmds=["python3", "frequency_par2.py"],
            image_pull_policy="IfNotPresent",
            env_vars=minio_env_vars,
            is_delete_operator_pod=True,
            node_selector={"node-role.kubernetes.io/worker": "worker"},
        ).expand(
            arguments=get_w_args(plan_data)
        )

        merger = KubernetesPodOperator.partial(
            task_id="merge",
            name=f"freq-merge-{pop.lower()}",
            namespace=NAMESPACE,
            image="kogsi/genome_dag:frequency_par2",
            cmds=["python3", "frequency_par2.py"],
            env_vars=minio_env_vars,
            is_delete_operator_pod=True,
            image_pull_policy="IfNotPresent",
            node_selector={"node-role.kubernetes.io/worker": "worker"},
        ).expand(
            arguments=get_m_args(plan_data)
        )

        feedback = report_feedback(plan_data, f"genome_frequency_{pop}", f"freq_{pop}", True)
        workers >> merger >> feedback


    # =================================
    # WIRING OF DAG
    # =================================
    individual_group = run_individual_tasks()

    # Sifting task
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
        node_selector={"node-role.kubernetes.io/worker": "worker"},
    )

    mutations_data = mutations_overlap_data(populations)
    mutations_tasks = KubernetesPodOperator.partial(
        task_id="mutations_overlap",
        name="mutations-overlap",
        namespace=NAMESPACE,
        image="kogsi/genome_dag:mutations-overlap",
        cmds=["python3", "mutations-overlap.py"],
        env_vars=minio_env_vars,
        get_logs=True,
        is_delete_operator_pod=True,
        image_pull_policy="IfNotPresent",
        node_selector={"kubernetes.io/worker": "worker"},
    ).expand(
        arguments=mutations_data
    )

    individual_group >> mutations_tasks
    sifting_task >> mutations_tasks

    for pop in populations:
        with TaskGroup(group_id=f"freq_{pop}") as frequency_group:
            run_frequency_tasks(pop)

        individual_group >> frequency_group
        sifting_task >> frequency_group
