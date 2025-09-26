from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Simple daily backup DAG placeholder. In production, replace BashOperator
# with Velero CLI calls or KubernetesPodOperator to run velero backup.
# TEST: Airflow can parse this DAG.
default_args = {
    'owner': 'cloudforge',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

dag = DAG(
    dag_id='cloudforge_daily_backup',
    default_args=default_args,
    description='Daily backup with Velero',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

validate_env = BashOperator(
    task_id='validate_environment',
    bash_command='echo "Validating environment..."',
    dag=dag,
)

run_backup = BashOperator(
    task_id='run_velero_backup',
    bash_command='echo "velero backup create cloudforge-$(date +%Y%m%d)"',
    dag=dag,
)

validate_env >> run_backup
