from airflow.sdk import task, dag
from datetime import datetime, timedelta

import pandas as pd
import os
import sys

sys.path.append("/usr/local/airflow/include")

from utils.sales_data_generator import SalesDataGenerator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 2, 7),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    dag_id="sales_forecast",
    default_args=default_args,
    description="Train sales forecasting models",
    schedule="@weekly",
    start_date=datetime(2026, 2, 7),
    catchup=False,
    tags=["training", "sales"],
)

def sales_forecast():
    @task()
    def extract_data():
        data_output_dir = "/tmp/sales_data" #stored in docker container (airflow scheduler), not persisted
        generator = SalesDataGenerator(
            start_date="2025-01-01", end_date="2026-01-01"
        )
        print("Generating realistic sales data...")
        file_paths = generator.generate_sales_data(output_dir=data_output_dir)
        total_files = sum(len(paths) for paths in file_paths.values())
        print(f"Generated {total_files} files:")
        for data_type, paths in file_paths.items():
            print(f"  - {data_type}: {len(paths)} files")

        return {
            "data_output_dir": data_output_dir,
            "file_paths": file_paths,
            "total_files": total_files,
        }

    @task()
    ## Note: In a real implementation, we would likely want to break this validation into multiple tasks for better modularity and error handling. 
    ## For demonstration, we are keeping it in one task.
    ## There are also many more validation checks we could implement (e.g. checking for duplicates, etc.)
    def validate_data(extracted_data):
        file_paths = extracted_data['file_paths']
        val_issues = []
        rows_checked = 0

        ## Validate sales files
        for sales_file in file_paths['sales'][:5]: #just validate first 5 sales files for demonstration
            df = pd.read_parquet(sales_file)

            required_columns = {'date', 'store_id', 'product_id', 'quantity_sold', 'revenue'}

            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                val_issues.append(f"File {sales_file} is missing columns: {', '.join(missing_columns)}")
            if df.empty:
                val_issues.append(f"File {sales_file} is empty.")
            if (df['revenue'] <0).any():
                val_issues.append(f"File {sales_file} has negative revenue values.")
            if (df['quantity_sold'] < 0).any():
                val_issues.append(f"File {sales_file} has negative quantity values.")
            rows_checked += len(df)

        for customer_file in file_paths['customer_traffic'][:5]:
            df = pd.read_parquet(customer_file)
            if df.empty:
                val_issues.append(f"File {customer_file} is empty.")
            if (df['customer_traffic'] < 0).any():
                val_issues.append(f"File {customer_file} has negative customer traffic values.")
            rows_checked += len(df)
        
        for data_category in ['store_events', 'promotions', 'inventory']: # for other categories just print the shape
            if data_category in file_paths and file_paths[data_category]: 
                test_file = file_paths[data_category][0]
                df = pd.read_parquet(test_file)
                print(f"{data_category} shape: {df.shape}")
                print(f"{data_category} columns: {df.columns.tolist()}")

        validation_report = {
            "rows_checked": rows_checked,
            "validation_errors": val_issues[:5], # just return first 5 issues for demonstration
            "issue_count": len(val_issues),
            "file_paths": file_paths
        }

        if val_issues:
            print(validation_report)
            for issue in val_issues:
                print(issue)
        else:
            print(validation_report)
        
        return validation_report

    extract_data_task = extract_data()
    validate_data_task = validate_data(extracted_data=extract_data_task)

sales_forecast_dag = sales_forecast()

    