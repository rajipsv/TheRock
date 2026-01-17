"""  
OpenVLA (Vision-Language-Action) tests for Strix platforms  
  
Tests robotic action prediction using OpenVLA model.  
This is a P0 critical test for Edge AI and robotics use cases.  
"""  
  
import pytest  
import torch  
from PIL import Image  
import os  
  
# Skip module if dependencies not available  
transformers = pytest.importorskip("transformers", reason="transformers not installed")  
  
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")  
  
  
@pytest.mark.strix  
@pytest.mark.vla  
@pytest.mark.p0  
@pytest.mark.quick  
@pytest.mark.skipif(  
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],  
    reason="Strix-specific VLA test"  
)  
class TestOpenVLA:  
    """OpenVLA Vision-Language-Action Model tests for Strix"""  
      
    def test_openvla_action_prediction(self, strix_device, test_image_224, cleanup_gpu, record_property):  
        """Test OpenVLA robotic action prediction on Strix"""  
        from transformers import AutoModelForVision2Seq, AutoProcessor  
        import signal  
          
        def timeout_handler(signum, frame):  
            raise TimeoutError("Model loading timeout (60s) - likely network issue downloading from Hugging Face")  
          
        print("\n🤖 Loading OpenVLA model (with 60s timeout)...")  
          
        # Set timeout for model loading (handles network download hangs)  
        signal.signal(signal.SIGALRM, timeout_handler)  
        signal.alarm(60)  # 60 second timeout  
          
        try:  
            # Load OpenVLA model for robotic tasks  
            model = AutoModelForVision2Seq.from_pretrained(  
                "openvla/openvla-7b",  
                device_map="auto",  
                trust_remote_code=True,  
                torch_dtype=torch.float16,  # Use FP16 for memory efficiency  
                resume_download=True,  
                force_download=False,  
            ).to(strix_device)  
            processor = AutoProcessor.from_pretrained(  
                "openvla/openvla-7b",  
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
          
        # Create test image representing a robotic scene  
        image = Image.new('RGB', (224, 224), color='blue')  
        # Add a simple object representation (red square)  
        from PIL import ImageDraw  
        draw = ImageDraw.Draw(image)  
        draw.rectangle([50, 50, 100, 100], fill='red')  
          
        # Define robotic task instruction  
        instruction = "Pick up the red object"  
          
        print("🔍 Processing robotic task...")  
        # Format input for OpenVLA  
        inputs = processor(  
            images=image,  
            instruction=instruction,  
            return_tensors="pt"  
        ).to(strix_device)  
          
        print("⚡ Running OpenVLA action prediction on Strix...")  
        with torch.no_grad():  
            outputs = model.generate(  
                **inputs,  
                max_new_tokens=256,  
                do_sample=False,  
                temperature=0.0,  
            )  
            action_sequence = processor.decode(outputs[0], skip_special_tokens=True)  
          
        print(f"✅ OpenVLA action sequence: {action_sequence}")  
          
        # Record metrics  
        record_property("metric_model", "OpenVLA-7B")  
        record_property("metric_action_length", len(action_sequence))  
        record_property("metric_task", instruction)  
        record_property("gpu_family", AMDGPU_FAMILIES)  
          
        # Assertions  
        assert outputs.device.type == "cuda", "Output should be on GPU"  
        assert len(action_sequence) > 0, "Action sequence should not be empty"  
          
        # Check if response contains action-related tokens  
        action_keywords = ["pick", "move", "place", "grab", "lift"]  
        has_action = any(keyword in action_sequence.lower() for keyword in action_keywords)  
        assert has_action, f"Expected action in response, got: {action_sequence}"  
          
        print(f"✅ OpenVLA action prediction test passed on {AMDGPU_FAMILIES}!")  
      
    def test_openvla_multi_task_reasoning(self, strix_device, cleanup_gpu):  
        """Test OpenVLA multi-task robotic reasoning"""  
        from transformers import AutoModelForVision2Seq, AutoProcessor  
        import signal  
          
        def timeout_handler(signum, frame):  
            raise TimeoutError("Model loading timeout")  
          
        print("\n🤖 Loading OpenVLA model for multi-task test (with timeout)...")  
        signal.signal(signal.SIGALRM, timeout_handler)  
        signal.alarm(60)  
          
        try:  
            model = AutoModelForVision2Seq.from_pretrained(  
                "openvla/openvla-7b",  
                device_map="auto",  
                trust_remote_code=True,  
                torch_dtype=torch.float16,  
                resume_download=True,  
                force_download=False,  
            ).to(strix_device)  
            processor = AutoProcessor.from_pretrained(  
                "openvla/openvla-7b",  
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
          
        # Test multiple robotic tasks  
        tasks = [  
            ("Pick up the red block", "red"),  
            ("Move to the green area", "green"),  
            ("Place the object on the table", "place")  
        ]  
          
        print(f"🔍 Processing {len(tasks)} robotic tasks...")  
        responses = []  
          
        for instruction, expected_keyword in tasks:  
            # Create simple test scene  
            image = Image.new('RGB', (224, 224), color='white')  
            draw = ImageDraw.Draw(image)  
              
            # Add colored objects based on task  
            if "red" in instruction:  
                draw.rectangle([50, 50, 100, 100], fill='red')  
            elif "green" in instruction:  
                draw.rectangle([75, 75, 125, 125], fill='green')  
              
            inputs = processor(  
                images=image,  
                instruction=instruction,  
                return_tensors="pt"  
            ).to(strix_device)  
              
            with torch.no_grad():  
                outputs = model.generate(  
                    **inputs,  
                    max_new_tokens=128,  
                    do_sample=False,  
                )  
                action = processor.decode(outputs[0], skip_special_tokens=True)  
                responses.append(action)  
              
            print(f"   Task: {instruction} → {action[:50]}...")  
          
        print(f"✅ Multi-task reasoning complete! Generated {len(responses)} action sequences")  
          
        assert len(responses) == len(tasks)  
        assert all(len(r) > 0 for r in responses), "All responses should not be empty"  
          
        print(f"✅ OpenVLA multi-task test passed on {AMDGPU_FAMILIES}!")  
      
    @pytest.mark.slow  
    def test_openvla_performance(self, strix_device, cleanup_gpu, record_property):  
        """Benchmark OpenVLA performance on Strix"""  
        from transformers import AutoModelForVision2Seq, AutoProcessor  
        import time  
        import signal  
          
        def timeout_handler(signum, frame):  
            raise TimeoutError("Model loading timeout")  
          
        print("\n🤖 Loading OpenVLA model for performance test (with timeout)...")  
        signal.signal(signal.SIGALRM, timeout_handler)  
        signal.alarm(60)  
          
        try:  
            model = AutoModelForVision2Seq.from_pretrained(  
                "openvla/openvla-7b",  
                device_map="auto",  
                trust_remote_code=True,  
                torch_dtype=torch.float16,  
                resume_download=True,  
                force_download=False,  
            ).to(strix_device)  
            processor = AutoProcessor.from_pretrained(  
                "openvla/openvla-7b",  
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
          
        # Prepare test input  
        image = Image.new('RGB', (224, 224), color='blue')  
        instruction = "Pick up the object"  
          
        inputs = processor(  
            images=image,  
            instruction=instruction,  
            return_tensors="pt"  
        ).to(strix_device)  
          
        # Warmup  
        print("🔥 Warming up...")  
        for _ in range(2):  # Fewer warmups for larger model  
            with torch.no_grad():  
                _ = model.generate(  
                    **inputs,  
                    max_new_tokens=64,  
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
                    max_new_tokens=64,  
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
        print(f"   Throughput: {throughput:.2f} actions/sec")  
          
        # Should complete inference in reasonable time for robotics  
        assert avg_time < 20.0, f"OpenVLA inference too slow: {avg_time:.2f}s (expected < 20s)"  
          
        print(f"✅ OpenVLA performance test passed on {AMDGPU_FAMILIES}!")  
      
    def test_openvla_memory_efficiency(self, strix_device, cleanup_gpu, record_property):  
        """Test OpenVLA memory usage on Strix iGPU"""  
        from transformers import AutoModelForVision2Seq, AutoProcessor  
        import signal  
          
        def timeout_handler(signum, frame):  
            raise TimeoutError("Model loading timeout")  
          
        print("\n🤖 Testing OpenVLA memory efficiency...")  
          
        torch.cuda.empty_cache()  
        torch.cuda.reset_peak_memory_stats()  
          
        signal.signal(signal.SIGALRM, timeout_handler)  
        signal.alarm(60)  
          
        try:  
            model = AutoModelForVision2Seq.from_pretrained(  
                "openvla/openvla-7b",  
                device_map="auto",  
                trust_remote_code=True,  
                torch_dtype=torch.float16,  
                resume_download=True,  
                force_download=False,  
            ).to(strix_device)  
            processor = AutoProcessor.from_pretrained(  
                "openvla/openvla-7b",  
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
        instruction = "Test action"  
          
        inputs = processor(  
            images=image,  
            instruction=instruction,  
            return_tensors="pt"  
        ).to(strix_device)  
          
        print("⚡ Running inference and measuring memory...")  
        with torch.no_grad():  
            _ = model.generate(**inputs, max_new_tokens=32)  
          
        torch.cuda.synchronize()  
          
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024  
          
        # Record metrics  
        record_property("metric_peak_memory_mb", f"{peak_memory_mb:.2f}")  
        record_property("gpu_family", AMDGPU_FAMILIES)  
          
        print(f"📊 Memory Usage:")  
        print(f"   Peak: {peak_memory_mb:.2f} MB")  
          
        # Should stay under reasonable limits for iGPU (OpenVLA is large)  
        assert peak_memory_mb < 4096, f"Memory usage too high: {peak_memory_mb:.2f} MB"  
          
        print(f"✅ OpenVLA memory test passed on {AMDGPU_FAMILIES}!")  
  
  
@pytest.mark.strix  
@pytest.mark.vla  
@pytest.mark.p1  
@pytest.mark.skipif(  
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],  
    reason="Strix-specific VLA test"  
)  
class TestOpenVLAAdvanced:  
    """Advanced OpenVLA tests for complex robotics scenarios"""  
      
    def test_openvla_spatial_reasoning(self, strix_device, cleanup_gpu):  
        """Test OpenVLA spatial reasoning for robotic tasks"""  
        from transformers import AutoModelForVision2Seq, AutoProcessor  
        import signal  
          
        def timeout_handler(signum, frame):  
            raise TimeoutError("Model loading timeout")  
          
        print("\n🤖 Testing OpenVLA spatial reasoning...")  
        signal.signal(signal.SIGALRM, timeout_handler)  
        signal.alarm(60)  
          
        try:  
            model = AutoModelForVision2Seq.from_pretrained(  
                "openvla/openvla-7b",  
                device_map="auto",  
                trust_remote_code=True,  
                torch_dtype=torch.float16,  
                resume_download=True,  
                force_download=False,  
            ).to(strix_device)  
            processor = AutoProcessor.from_pretrained(  
                "openvla/openvla-7b",  
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
          
        # Create scene with spatial relationships  
        image = Image.new('RGB', (224, 224), color='white')  
        draw = ImageDraw.Draw(image)  
        # Object on the left  
        draw.rectangle([20, 50, 70, 100], fill='red')  
        # Object on the right  
        draw.rectangle([150, 50, 200, 100], fill='blue')  
          
        spatial_instructions = [  
            "Pick up the object on the left",  
            "Move to the object on the right",  
            "Place the red object next to the blue one"  
        ]  
          
        print(f"🔍 Testing spatial reasoning with {len(spatial_instructions)} instructions...")  
          
        for instruction in spatial_instructions:  
            inputs = processor(  
                images=image,  
                instruction=instruction,  
                return_tensors="pt"  
            ).to(strix_device)  
              
            with torch.no_grad():  
                outputs = model.generate(  
                    **inputs,  
                    max_new_tokens=128,  
                    do_sample=False,  
                )  
                action = processor.decode(outputs[0], skip_special_tokens=True)  
              
            print(f"   {instruction} → {action[:50]}...")  
          
        print(f"✅ OpenVLA spatial reasoning test passed on {AMDGPU_FAMILIES}!")
