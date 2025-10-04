#!/usr/bin/env python3
"""
CloudForge AI - MinIO Storage Upload Test Script
Uploads Kaggle CSV datasets to distributed MinIO storage cluster
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import json
import argparse
from pathlib import Path

# MinIO client configuration
try:
    from minio import Minio
    from minio.error import ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists
except ImportError:
    print("Installing required packages...")
    os.system("pip install minio pandas requests")
    from minio import Minio
    from minio.error import ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists

class CloudForgeStorageUploader:
    """
    Handles uploading of test datasets to CloudForge AI MinIO storage
    """
    
    def __init__(self, endpoint: str = "localhost:9000", 
                 access_key: str = "cloudforge-admin", 
                 secret_key: str = "CloudForge2025!"):
        """
        Initialize MinIO client for CloudForge AI storage
        
        Args:
            endpoint: MinIO server endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = "cloudforge-datasets"
        
        # Initialize MinIO client
        try:
            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=False  # Set to True for HTTPS
            )
            print(f"✅ Connected to MinIO at {endpoint}")
        except Exception as e:
            print(f"❌ Failed to connect to MinIO: {e}")
            sys.exit(1)
    
    def create_bucket_if_not_exists(self) -> bool:
        """
        Create the CloudForge datasets bucket if it doesn't exist
        
        Returns:
            bool: True if bucket exists or was created successfully
        """
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"✅ Created bucket: {self.bucket_name}")
                
                # Set bucket policy for read access
                policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": "*"},
                            "Action": ["s3:GetObject"],
                            "Resource": [f"arn:aws:s3:::{self.bucket_name}/*"]
                        }
                    ]
                }
                self.client.set_bucket_policy(self.bucket_name, json.dumps(policy))
                print(f"✅ Set public read policy for bucket: {self.bucket_name}")
            else:
                print(f"✅ Bucket already exists: {self.bucket_name}")
            return True
        except BucketAlreadyOwnedByYou:
            print(f"✅ Bucket already owned: {self.bucket_name}")
            return True
        except BucketAlreadyExists:
            print(f"✅ Bucket already exists: {self.bucket_name}")
            return True
        except Exception as e:
            print(f"❌ Error creating bucket: {e}")
            return False
    
    def upload_file(self, file_path: str, object_name: Optional[str] = None) -> bool:
        """
        Upload a file to MinIO storage
        
        Args:
            file_path: Path to the file to upload
            object_name: Name of the object in storage (default: filename)
            
        Returns:
            bool: True if upload successful
        """
        if object_name is None:
            object_name = os.path.basename(file_path)
        
        try:
            # Add metadata
            metadata = {
                "uploaded-by": "cloudforge-ai",
                "upload-time": datetime.now().isoformat(),
                "file-size": str(os.path.getsize(file_path))
            }
            
            # Upload file
            self.client.fput_object(
                self.bucket_name,
                object_name,
                file_path,
                metadata=metadata
            )
            print(f"✅ Uploaded: {file_path} → {object_name}")
            return True
        except Exception as e:
            print(f"❌ Upload failed for {file_path}: {e}")
            return False
    
    def upload_csv_with_validation(self, file_path: str, object_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload CSV file with data validation and metadata extraction
        
        Args:
            file_path: Path to the CSV file
            object_name: Name of the object in storage
            
        Returns:
            dict: Upload result with metadata
        """
        if object_name is None:
            object_name = f"datasets/{os.path.basename(file_path)}"
        
        try:
            # Validate CSV file
            df = pd.read_csv(file_path)
            
            # Extract metadata
            metadata = {
                "file-type": "csv",
                "rows": str(len(df)),
                "columns": str(len(df.columns)),
                "column-names": ",".join(df.columns.tolist()),
                "file-size-mb": str(round(os.path.getsize(file_path) / 1024 / 1024, 2)),
                "uploaded-by": "cloudforge-ai",
                "upload-time": datetime.now().isoformat(),
                "data-types": ",".join(df.dtypes.astype(str).tolist())
            }
            
            # Upload with metadata
            self.client.fput_object(
                self.bucket_name,
                object_name,
                file_path,
                metadata=metadata
            )
            
            result = {
                "success": True,
                "object_name": object_name,
                "metadata": metadata,
                "preview": df.head().to_dict()
            }
            
            print(f"✅ CSV Upload successful: {object_name}")
            print(f"   📊 Rows: {metadata['rows']}, Columns: {metadata['columns']}")
            print(f"   📁 Size: {metadata['file-size-mb']} MB")
            
            return result
            
        except Exception as e:
            print(f"❌ CSV upload failed for {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def list_uploaded_files(self) -> list:
        """
        List all uploaded files in the bucket
        
        Returns:
            list: List of object information
        """
        try:
            objects = self.client.list_objects(self.bucket_name, recursive=True)
            file_list = []
            
            for obj in objects:
                # Get object metadata
                try:
                    obj_stat = self.client.stat_object(self.bucket_name, obj.object_name)
                    file_info = {
                        "name": obj.object_name,
                        "size": obj_stat.size,
                        "last_modified": obj_stat.last_modified,
                        "metadata": obj_stat.metadata
                    }
                    file_list.append(file_info)
                except Exception:
                    # If metadata retrieval fails, add basic info
                    file_list.append({
                        "name": obj.object_name,
                        "size": obj.size,
                        "last_modified": obj.last_modified
                    })
            
            return file_list
        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return []
    
    def download_sample_datasets(self) -> list:
        """
        Download sample Kaggle-style datasets for testing
        
        Returns:
            list: List of downloaded dataset files
        """
        datasets = []
        
        # Sample dataset URLs (using public datasets)
        sample_datasets = [
            {
                "name": "iris_dataset.csv",
                "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
                "description": "Classic Iris flower dataset"
            },
            {
                "name": "tips_dataset.csv", 
                "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
                "description": "Restaurant tips dataset"
            },
            {
                "name": "flights_dataset.csv",
                "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
                "description": "Flight passenger data"
            }
        ]
        
        # Create temp directory for downloads
        temp_dir = Path("temp_datasets")
        temp_dir.mkdir(exist_ok=True)
        
        for dataset in sample_datasets:
            try:
                print(f"📥 Downloading {dataset['name']}...")
                response = requests.get(dataset['url'])
                response.raise_for_status()
                
                file_path = temp_dir / dataset['name']
                with open(file_path, 'w') as f:
                    f.write(response.text)
                
                datasets.append(str(file_path))
                print(f"✅ Downloaded: {dataset['name']} - {dataset['description']}")
                
            except Exception as e:
                print(f"❌ Failed to download {dataset['name']}: {e}")
        
        return datasets

def main():
    """
    Main function to run the upload test
    """
    parser = argparse.ArgumentParser(description="CloudForge AI MinIO Storage Upload Test")
    parser.add_argument("--endpoint", default="localhost:9000", help="MinIO endpoint")
    parser.add_argument("--minikube", action="store_true", help="Use minikube IP (auto-detect)")
    parser.add_argument("--upload-port", default="4000", help="Upload service port")
    parser.add_argument("--file", help="Specific file to upload")
    parser.add_argument("--download-samples", action="store_true", help="Download sample datasets")
    
    args = parser.parse_args()
    
    # Auto-detect minikube IP if requested
    if args.minikube:
        try:
            import subprocess
            result = subprocess.run(['minikube', 'ip'], capture_output=True, text=True)
            if result.returncode == 0:
                minikube_ip = result.stdout.strip()
                args.endpoint = f"{minikube_ip}:30900"  # NodePort for MinIO
                print(f"🔍 Detected minikube IP: {minikube_ip}")
        except Exception as e:
            print(f"⚠️  Could not detect minikube IP: {e}")
    
    # Initialize uploader
    uploader = CloudForgeStorageUploader(endpoint=args.endpoint)
    
    # Create bucket
    if not uploader.create_bucket_if_not_exists():
        print("❌ Failed to create/access bucket")
        sys.exit(1)
    
    uploaded_files = []
    
    # Download sample datasets if requested
    if args.download_samples:
        print("\n📥 Downloading sample datasets...")
        datasets = uploader.download_sample_datasets()
        
        # Upload downloaded datasets
        for dataset_file in datasets:
            result = uploader.upload_csv_with_validation(dataset_file)
            if result["success"]:
                uploaded_files.append(result)
    
    # Upload specific file if provided
    if args.file:
        if os.path.exists(args.file):
            if args.file.endswith('.csv'):
                result = uploader.upload_csv_with_validation(args.file)
                if result["success"]:
                    uploaded_files.append(result)
            else:
                if uploader.upload_file(args.file):
                    uploaded_files.append({"object_name": os.path.basename(args.file)})
        else:
            print(f"❌ File not found: {args.file}")
    
    # If no specific actions, show help and list existing files
    if not args.download_samples and not args.file:
        print("\n📁 Current files in storage:")
        files = uploader.list_uploaded_files()
        if files:
            for file_info in files:
                print(f"  📄 {file_info['name']} ({file_info.get('size', 'unknown')} bytes)")
        else:
            print("  📂 No files found")
        
        print("\n💡 Usage examples:")
        print(f"  python {sys.argv[0]} --download-samples")
        print(f"  python {sys.argv[0]} --file dataset.csv")
        print(f"  python {sys.argv[0]} --minikube --download-samples")
        print(f"  curl -X POST http://minikube-ip:{args.upload_port}/storage/upload -F 'file=@dataset.csv'")
    
    # Summary
    if uploaded_files:
        print(f"\n🎉 Upload Summary: {len(uploaded_files)} files uploaded successfully")
        for file_info in uploaded_files:
            print(f"  ✅ {file_info['object_name']}")
        
        print(f"\n🔗 Access your files at: http://{args.endpoint}")
        print(f"🎛️  MinIO Console: http://{args.endpoint.replace(':9000', ':9001')}")
    
    # Test storage/upload endpoint if available
    try:
        upload_endpoint = f"http://{args.endpoint.split(':')[0]}:{args.upload_port}/storage/upload"
        response = requests.get(upload_endpoint.replace('/upload', '/health'))
        if response.status_code == 200:
            print(f"\n🌐 Storage API available at: {upload_endpoint}")
        else:
            print(f"\n⚠️  Storage API not available at: {upload_endpoint}")
    except Exception:
        print(f"\n⚠️  Could not test storage API at: http://{args.endpoint.split(':')[0]}:{args.upload_port}/storage/upload")

if __name__ == "__main__":
    main()