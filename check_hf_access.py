"""
check_hf_access.py
==================
Diagnostic script to check HuggingFace access and Llama 3.1 permissions
"""

import sys
from huggingface_hub import HfApi, whoami
from huggingface_hub.utils import HfHubHTTPError

print("="*80)
print("HUGGINGFACE ACCESS DIAGNOSTIC")
print("="*80)

# Step 1: Check if logged in
print("\n1️⃣  Checking HuggingFace login status...")
try:
    user_info = whoami()
    print(f"✅ Logged in as: {user_info['name']}")
    print(f"   Account type: {user_info.get('type', 'N/A')}")
    token_available = True
except Exception as e:
    print(f"❌ Not logged in: {str(e)}")
    print("\n💡 Solution: Run this command:")
    print("   huggingface-cli login")
    token_available = False

# Step 2: Check Llama 3.1 access
if token_available:
    print("\n2️⃣  Checking Llama 3.2 access...")
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    try:
        api = HfApi()
        model_info = api.model_info(model_id)
        print(f"✅ Can access {model_id}")
        print(f"   Model ID: {model_info.modelId}")
        print(f"   Downloads: {model_info.downloads:,}")
        
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            print(f"❌ Access denied to {model_id}")
            print("\n💡 Solution:")
            print("   1. Visit: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
            print("   2. Click 'Agree and access repository'")
            print("   3. Wait a few minutes for access to be granted")
            print("   4. Re-run this script")
        elif e.response.status_code == 404:
            print(f"❌ Model not found: {model_id}")
            print("   This might be a typo in the model name")
        else:
            print(f"❌ Error accessing model: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

# Step 3: Check token in environment
print("\n3️⃣  Checking environment variables...")
import os
if "HF_TOKEN" in os.environ:
    print("✅ HF_TOKEN found in environment")
elif "HUGGING_FACE_HUB_TOKEN" in os.environ:
    print("✅ HUGGING_FACE_HUB_TOKEN found in environment")
else:
    print("ℹ️  No HF token in environment (this is fine if you used 'huggingface-cli login')")

# Step 4: Test tokenizer loading
if token_available:
    print("\n4️⃣  Testing tokenizer loading...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            token=True
        )
        print("✅ Tokenizer loaded successfully!")
        print(f"   Vocab size: {tokenizer.vocab_size:,}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {str(e)}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if not token_available:
    print("\n❌ You need to login to HuggingFace")
    print("\nRun these commands:")
    print("  1. huggingface-cli login")
    print("  2. python check_hf_access.py  # Run this script again")
else:
    print("\n✅ HuggingFace access is configured")
    print("\nIf you still have issues:")
    print("  1. Make sure you accepted the Llama 3.1 license")
    print("  2. Wait 5-10 minutes after accepting")
    print("  3. Try logging in again: huggingface-cli login")
    print("  4. Check your internet connection")

print("="*80)