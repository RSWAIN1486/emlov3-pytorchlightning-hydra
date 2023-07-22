import boto3

def get_aws_credentials():
    print("Please enter your AWS credentials:")
    aws_access_key_id = input("Access Key ID: ")
    aws_secret_access_key = input("Secret Access Key: ")
    # aws_region = 'input("AWS Region: ")'
    aws_region = 'ap-south-1'
    return aws_access_key_id, aws_secret_access_key, aws_region

def main():
    # aws_access_key_id, aws_secret_access_key, aws_region = get_aws_credentials()

    # # DO THIS IF YOU DONT HAVE AWS CONFIGURED - Create a Boto3 session with the provided credentials 
    # session = boto3.Session(
    #     aws_access_key_id=aws_access_key_id,
    #     aws_secret_access_key=aws_secret_access_key,
    #     region_name=aws_region
    # )

    # Now you can create any AWS service client using this session
    # s3 = session.client('s3')
    s3 = boto3.client('s3')

    # Upload model to S3
    bucket_name = 'emlov3-raks'
    model_file_path = './ckpt/gpt_torch_script.pt'
    s3.upload_file(model_file_path, bucket_name, 'gpt_scripted_in_s3_bucket/gpt_torch_script.pt')

if __name__ == "__main__":
    main()
