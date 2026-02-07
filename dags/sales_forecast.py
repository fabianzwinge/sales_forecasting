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

    @task()
    def clean_data(validated_data):
        file_paths = validated_data['file_paths']

        #clean sales data
        sales_dfs = []
        for sales_file in file_paths['sales'][:100]: #just read first 100 sales files for demonstration
            df = pd.read_parquet(sales_file)
            sales_dfs.append(df)
        sales_data = pd.concat(sales_dfs, ignore_index=True)
        daily_sales = (
            sales_data.groupby(['date', 'store_id', 'product_id', 'category'])
            .agg({'quantity_sold': 'sum', 'revenue': 'sum', 'cost': 'sum', 'discount_percent': 'sum', 'profit': 'sum'})
            .reset_index()
            .sort_values('date')
        )

        daily_sales = daily_sales.rename(columns={'revenue': 'total_revenue', 'quantity_sold': 'total_quantity', 'cost': 'total_cost', 'discount_percent': 'total_discount_percent', 'profit': 'total_profit'})
        daily_sales['date'] = pd.to_datetime(daily_sales['date'])

        #clean promotions data and merge with sales
        if file_paths['promotions']:
            promo_df = pd.read_parquet(file_paths['promotions'][0])
            promo_summary = (
                promo_df.groupby(['date', 'product_id'])['discount_percent']
                .max()
                .reset_index()
            )

            promo_summary['has_discount'] = 1

            daily_sales = daily_sales.merge(
                promo_summary[['date', 'product_id', 'has_discount']],
                on=['date', 'product_id'],
                how='left'
            )
            daily_sales['has_discount'] = daily_sales['has_discount'].fillna(0).astype(int)
        
        #clean customer traffic data and merge with sales
        if file_paths['customer_traffic']:
            traffic_dfs = []
            for traffic_file in file_paths['customer_traffic'][:100]:
                df = pd.read_parquet(traffic_file)
                traffic_dfs.append(df)
            traffic_data = pd.concat(traffic_dfs, ignore_index=True)

            traffic_summary = (
                traffic_data.groupby(['date', 'store_id'])
                .agg({'customer_traffic': 'sum', 'is_holiday': 'max'})
                .reset_index()
            )

            daily_sales = daily_sales.merge(
                traffic_summary,
                on=['date', 'store_id'],
                how='left'
            )
            daily_sales['customer_traffic'] = daily_sales['customer_traffic'].fillna(daily_sales['customer_traffic'].median()).astype(int)
            
            # aggregate to store level for modeling
            store_daily_sales = (
                daily_sales.groupby(['date', 'store_id'])
                .agg({'has_discount': 'mean', 'total_revenue': 'sum', 'total_quantity': 'sum', 'customer_traffic': 'first', 'is_holiday': 'first'})
                .reset_index()
            )
            
            print("Data cleaning complete. Sample of cleaned data:")
            print(store_daily_sales.head())

            return store_daily_sales


    @task()
    def train_models(cleaned_data):
        print("Training models with cleaned data...")
        print(f"Cleaned data shape: {cleaned_data.shape}")
        print(f"Cleaned data columns: {cleaned_data.columns.tolist()}")
        print(cleaned_data.head())

    extract_data_task = extract_data()
    validate_data_task = validate_data(extracted_data=extract_data_task)
    clean_data_task = clean_data(validated_data=validate_data_task)
    train_models_task = train_models(cleaned_data=clean_data_task)

sales_forecast_dag = sales_forecast()

    