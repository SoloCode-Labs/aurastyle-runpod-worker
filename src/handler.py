import os
import urllib.parse
import boto3
import runpod
from PIL import Image

# Initialize boto3 S3 client
# AWS credentials will be read from environment variables on RunPod
s3_client = boto3.client("s3")

def parse_s3_uri(s3_uri: str):
    """
    Parses an S3 URI (e.g., s3://bucket-name/path/to/file.jpg)
    and returns (bucket_name, key).
    """
    parsed = urllib.parse.urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URI scheme: {s3_uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key

def download_from_s3(s3_uri: str, local_path: str):
    bucket, key = parse_s3_uri(s3_uri)
    print(f"Downloading from S3: bucket={bucket}, key={key} to {local_path}")
    s3_client.download_file(bucket, key, local_path)

def upload_to_s3(local_path: str, s3_uri: str):
    bucket, key = parse_s3_uri(s3_uri)
    print(f"Uploading to S3: {local_path} to bucket={bucket}, key={key}")
    s3_client.upload_file(local_path, bucket, key)

def process_hair_simulation(input_image_path: str, prompt: str, negative_prompt: str, strength: float, output_path: str):
    """
    Main AI hairstyle generation function.
    In a real production environment, this imports Diffusers and runs InstantID.
    """
    print(f"Starting hairstyle simulation with prompt: '{prompt}', strength: {strength}")
    
    # Load input image
    image = Image.open(input_image_path)
    
    # [Aura Style AI - Hair Simulation Engine Placeholder]
    # In a real deployment:
    # 1. Load ControlNet/InstantID pipelines
    # 2. Run pipe(prompt=prompt, image=image, negative_prompt=negative_prompt, controlnet_conditioning_scale=strength)
    # For boilerplate demonstration, we apply a placeholder crop/filter to verify pipeline connection
    
    # Simple image manipulation placeholder (convert to RGB, resize slightly or save)
    processed_image = image.convert("RGB")
    processed_image.save(output_path, "JPEG")
    print(f"Generation successful. Saved output to {output_path}")

def handler(job):
    """
    RunPod Serverless Handler
    """
    # Parse inputs from the request body
    job_input = job["input"]
    input_image_uri = job_input.get("input_image")
    prompt = job_input.get("prompt", "classic fade haircut")
    negative_prompt = job_input.get("negative_prompt", "")
    instantid_strength = job_input.get("instantid_strength", 0.8)
    output_bucket_uri = job_input.get("output_bucket")

    if not input_image_uri or not output_bucket_uri:
        return {"error": "Missing input_image or output_bucket in job payload"}

    # Define temporary files
    local_input = "/tmp/input.jpg"
    local_output = "/tmp/output.jpg"

    try:
        # 1. Download input selfie from temporal S3 bucket
        download_from_s3(input_image_uri, local_input)

        # 2. Run hairstyle simulation
        process_hair_simulation(
            input_image_path=local_input,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=instantid_strength,
            output_path=local_output
        )

        # 3. Upload generated output back to temporal S3 bucket
        upload_to_s3(local_output, output_bucket_uri)

        # Cleanup temp files
        if os.path.exists(local_input):
            os.remove(local_input)
        if os.path.exists(local_output):
            os.remove(local_output)

        return {"status": "COMPLETED"}

    except Exception as e:
        print(f"Error executing RunPod job: {str(e)}")
        return {"error": f"Inference execution failed: {str(e)}"}

# Start RunPod serverless service
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
