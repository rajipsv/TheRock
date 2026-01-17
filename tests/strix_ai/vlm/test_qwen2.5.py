"""  
Qwen2.5-VL (Vision Language) tests for Strix platforms  
  
Tests vision-language understanding using Alibaba's Qwen2.5-VL model.  
This is a P0 critical test for Edge AI and Windows Copilot+ use cases.  
"""  
  
import pytest  
import torch  
from PIL import Image  
import os  
  
# Skip module if dependencies not available  
transformers = pytest.importorskip("transformers", reason="transformers not installed")  
  
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")  
  
  
@pytest.mark.strix  
@pytest.mark.vlm  
@pytest.mark.p0  
@pytest.mark.quick  
@pytest.mark.skipif(  
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],  
    reason="Strix-specific VLM test"  
)  
class TestQwen25:  
    """Qwen2.5 Vision-Language Model tests for Strix"""  
      
    def test_qwen25_image_text_matching(self, strix_device, test_image_224, cleanup_gpu, record_property):  
        """Test Qwen2.5-VL image-text matching on Strix"""  
        from transformers import AutoModelForCausalLM, AutoTokenizer  
        import signal  
          
        def timeout_handler(signum, frame):  
            raise TimeoutError("Model loading timeout (60s) - likely network issue downloading from Hugging Face")  
          
        print("\n🧠 Loading Qwen2.5-VL model (with 60s timeout)...")  
          
        # Set timeout for model loading (handles network download hangs)  
        signal.signal(signal.SIGALRM, timeout_handler)  
        signal.alarm(60)  # 60 second timeout  
          
        try:  
            # Try to load from cache first, download if needed  
            model = AutoModelForCausalLM.from_pretrained(  
                "Qwen/Qwen2.5-VL-7B-Instruct",  
                device_map="auto",  
                trust_remote_code=True,  
                torch_dtype=torch.float16,  # Use FP16 for memory efficiency  
                resume_download=True,  
                force_download=False,  
            ).to(strix_device)  
            tokenizer = AutoTokenizer.from_pretrained(  
                "Qwen/Qwen2.5-VL-7B-Instruct",  
                trust_remote_code=True,  
                resume_download=True,  
                force_download=False,  
            )  
            signal.alarm(0)  # Cancel timeout  
            print("✅ Model loaded successfully")  
        except TimeoutError as e:  
            signal.alarm(0)  
            pytest.skip(f"Model loading timeout: {e}")  
        except Exception as e:  
            signal.alarm(0)  
            print(f"❌ Model loading failed: {e}")  
            raise  
          
        model.eval()  
          
        # Create test image  
        image = Image.new('RGB', (224, 224), color='red')  
        query = "What color is this image?"  
          
        print("🔍 Processing inputs...")  
        # Format input for Qwen2.5-VL (uses chat template)  
        messages = [  
            {  
                "role": "user",  
                "content": [  
                    {"type": "image", "image": image},  
                    {"type": "text", "text": query},  
                ],  
            }  
        ]  
          
        # Apply chat template  
        text = tokenizer.apply_chat_template(  
            messages,   
            tokenize=False,   
            add_generation_prompt=True  
        )  
          
        inputs = tokenizer(  
            text,   
            images=[image],   
            return_tensors="pt"  
        ).to(strix_device)  
          
        print("⚡ Running Qwen2.5-VL inference on Strix...")  
        with torch.no_grad():  
            outputs = model.generate(  
                **inputs,   
                max_new_tokens=128,  
                do_sample=False,  
                temperature=0.0,  
            )  
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)  
          
        print(f"✅ Qwen2.5-VL response: {response}")  
          
        # Record metrics  
        record_property("metric_model", "Qwen2.5-VL-7B-Instruct")  
        record_property("metric_response_length", len(response))  
        record_property("metric_prediction_correct", "true" if "red" in response.lower() else "false")  
        record_property("gpu_family", AMDGPU_FAMILIES)  
          
        # Assertions  
        assert outputs.device.type == "cuda", "Output should be on GPU"  
        assert len(response) > 0, "Response should not be empty"  
          
        # Check if model correctly identifies the red color  
        assert "red" in response.lower(), f"Expected response to contain 'red', got: {response}"  
          
        print(f"✅ Qwen2.5-VL test passed on {AMDGPU_FAMILIES}!")  
      
    def test_qwen25_batch_inference(self, strix_device, cleanup_gpu):  
        """Test Qwen2.5-VL batch processing on Strix"""  
        from transformers import AutoModelForCausalLM, AutoTokenizer  
        import signal  
          
        def timeout_handler(signum, frame):  
            raise TimeoutError("Model loading timeout")  
          
        print("\n🧠 Loading Qwen2.5-VL model for batch test (with timeout)...")  
        signal.signal(signal.SIGALRM, timeout_handler)  
        signal.alarm(60)  
          
        try:  
            model = AutoModelForCausalLM.from_pretrained(  
                "Qwen/Qwen2.5-VL-7B-Instruct",  
                device_map="auto",  
                trust_remote_code=True,  
                torch_dtype=torch.float16,  
                resume_download=True,  
                force_download=False,  
            ).to(strix_device)  
            tokenizer = AutoTokenizer.from_pretrained(  
                "Qwen/Qwen2.5-VL-7B-Instruct",  
                trust_remote_code=True,  
                resume_download=True,  
                force_download=False,  
            )  
            signal.alarm(0)  
        except TimeoutError as e:  
            signal.alarm(0)  
            pytest.skip(f"Model loading timeout: {e}")  
        except Exception as e:  
            signal.alarm(0)  
            raise  
          
        model.eval()  
          
        # Create batch of test images  
        batch_size = 1  # Qwen2.5-VL is memory intensive, use single image  
        images = [  
            Image.new('RGB', (224, 224), color='blue'),  
        ]  
        queries = ["What color is this image?"]  
          
        print(f"🔍 Processing {batch_size} image(s)...")  
        responses = []  
          
        for image, query in zip(images, queries):  
            messages = [  
                {  
                    "role": "user",  
                    "content": [  
                        {"type": "image", "image": image},  
                        {"type": "text", "text": query},  
                    ],  
                }  
            ]  
              
            text = tokenizer.apply_chat_template(  
                messages,   
                tokenize=False,   
                add_generation_prompt=True  
            )  
              
            inputs = tokenizer(  
                text,   
                images=[image],   
                return_tensors="pt"  
            ).to(strix_device)  
              
            with torch.no_grad():  
                outputs = model.generate(  
                    **inputs,   
                    max_new_tokens=64,  
                    do_sample=False,  
                )  
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)  
                responses.append(response)  
          
        print(f"✅ Inference successful! Generated {len(responses)} response(s)")  
          
        assert len(responses) == batch_size  
        assert all(len(r) > 0 for r in responses), "All responses should not be empty"  
          
        print(f"✅ Qwen2.5-VL batch test passed on {AMDGPU_FAMILIES}!")  
      
    @pytest.mark.slow  
    def test_qwen25_performance(self, strix_device, cleanup_gpu, record_property):  
        """Benchmark Qwen2.5-VL performance on Strix"""  
        from transformers import AutoModelForCausalLM, AutoTokenizer  
        import time  
        import signal  
          
        def timeout_handler(signum, frame):  
            raise TimeoutError("Model loading timeout")  
          
        print("\n🧠 Loading Qwen2.5-VL model for performance test (with timeout)...")  
        signal.signal(signal.SIGALRM, timeout_handler)  
        signal.alarm(60)  
          
        try:  
            model = AutoModelForCausalLM.from_pretrained(  
                "Qwen/Qwen2.5-VL-7B-Instruct",  
                device_map="auto",  
                trust_remote_code=True,  
                torch_dtype=torch.float16,  
                resume_download=True,  
                force_download=False,  
            ).to(strix_device)  
            tokenizer = AutoTokenizer.from_pretrained(  
                "Qwen/Qwen2.5-VL-7B-Instruct",  
                trust_remote_code=True,  
                resume_download=True,  
                force_download=False,  
            )  
            signal.alarm(0)  
        except TimeoutError as e:  
            signal.alarm(0)  
            pytest.skip(f"Model loading timeout: {e}")  
        except Exception as e:  
            signal.alarm(0)  
            raise  
          
        model.eval()  
          
        image = Image.new('RGB', (224, 224))  
        query = "Describe this image briefly"  
          
        messages = [  
            {  
                "role": "user",  
                "content": [  
                    {"type": "image", "image": image},  
                    {"type": "text", "text": query},  
                ],  
            }  
        ]  
          
        text = tokenizer.apply_chat_template(  
            messages,   
            tokenize=False,   
            add_generation_prompt=True  
        )  
          
        inputs = tokenizer(  
            text,   
            images=[image],   
            return_tensors="pt"  
        ).to(strix_device)  
          
        # Warmup  
        print("🔥 Warming up...")  
        for _ in range(2):  # Fewer warmups for larger model  
            with torch.no_grad():  
                _ = model.generate(  
                    **inputs,   
                    max_new_tokens=32,  
                    do_sample=False,  
                )  
        torch.cuda.synchronize()  
          
        # Benchmark  
        print("⏱️  Benchmarking...")  
        num_iterations = 5  # Fewer iterations for larger model  
        start = time.time()  
          
        for _ in range(num_iterations):  
            with torch.no_grad():  
                _ = model.generate(  
                    **inputs,   
                    max_new_tokens=32,  
                    do_sample=False,  
                )  
          
        torch.cuda.synchronize()  
        elapsed = time.time() - start  
          
        avg_time = elapsed / num_iterations  
        throughput = num_iterations / elapsed  
          
        # Record metrics  
        record_property("metric_avg_inference_time_s", f"{avg_time:.2f}")  
        record_property("metric_throughput_per_sec", f"{throughput:.2f}")  
        record_property("metric_iterations", num_iterations)  
        record_property("gpu_family", AMDGPU_FAMILIES)  
          
        print(f"📊 Performance Results:")  
        print(f"   Avg inference time: {avg_time:.2f} s")  
        print(f"   Throughput: {throughput:.2f} inferences/sec")  
          
        # Should complete inference in reasonable time for edge device  
        assert avg_time < 15.0, f"Qwen2.5-VL inference too slow: {avg_time:.2f}s (expected < 15s)"  
          
        print(f"✅ Qwen2.5-VL performance test passed on {AMDGPU_FAMILIES}!")
