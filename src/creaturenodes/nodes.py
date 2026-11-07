from google import genai
from google.genai import types
from PIL import Image
import io
from io import BytesIO
import numpy as np
from torchvision import transforms
import requests
import base64

modelEnum = {
    "Gemini 3 Flash Preview": "gemini-3-flash-preview",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "Imagen 4": "imagen-4.0-generate-001",
    "Gemini 2.5 Flash Image": "gemini-2.5-flash-image",
    "Gemini 3 Pro Image Preview": "gemini-3-pro-image-preview"
}

"""
Google API Nodes to image generation
"""

class GoogleGenAIClient:
    CATEGORY = "CreatureNodes/Google"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                })
            }
        }
    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("google_genai_client",)
    FUNCTION = "google_genai_client"
    
    def google_genai_client(self, api_key):
        
        if not api_key:
            raise RuntimeError("Provide an api key")
        
        try:
            client = genai.Client(api_key=api_key)
            return (client,)
        except Exception as e:
            raise RuntimeError(f"Failed to configure Google GenAI API: {e}")

class GoogleT2T:
    CATEGORY = "CreatureNodes/Google"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "google_genai_client": ("ANY",),
                "model": (["Gemini 3 Flash Preview", "Gemini 2.5 Pro", "Gemini 2.5 Flash", "Gemini 2.0 Flash"],),
                "system_prompt": ("STRING", {
                    "multiline": True
                }),
                "user_prompt": ("STRING", {
                    "multiline": True
                })
            }
        }
    RETURN_TYPES = ("STRING", "ANY",)
    RETURN_NAMES = ("generated_string", "google_genai_client,")
    FUNCTION = "google_t2t"
    
    def google_t2t(self, google_genai_client, model, system_prompt, user_prompt):
        try:
            response = google_genai_client.models.generate_content(
                model=modelEnum[model],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                ),
                contents=user_prompt
            )
            return (response.text, google_genai_client,)
        except Exception as e:
            raise RuntimeError(f"Failed to generate text: {e.message}")

class GoogleT2I:
    CATEGORY = "CreatureNodes/Google"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "google_genai_client": ("ANY",),
                "model": (["Imagen 4", "Gemini 2.5 Flash Image", "Gemini 3 Pro Image Preview"],),
                "aspect_ratio": (["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"],),
                "image_size": (["1K", "2K", "4K"],),
                "prompt": ("STRING", {
                    "multiline": True
                }),
            }
        }
    RETURN_TYPES = ("IMAGE", "ANY")
    RETURN_NAMES = ("generated_image", "google_genai_client")
    FUNCTION = "google_t2i"
    
    def google_t2i(self, google_genai_client, model, aspect_ratio, image_size, prompt):
        try:
            if model in ["Imagen 4"]:
                response = google_genai_client.models.generate_images(
                    model=modelEnum[model],
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        image_size=image_size,
                        aspect_ratio=aspect_ratio
                    )
                )

                # Extract raw bytes from generated_images
                image_bytes = response.generated_images[0].image.image_bytes

                # Convert bytes → PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")

            else:
                # Use generate_content for other models
                response = google_genai_client.models.generate_content(
                    model=modelEnum[model],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                            image_size=image_size
                        )
                    )
                )

                # Extract the inline_data
                image_parts = [part for part in response.parts if part.inline_data]
                image_bytes = image_parts[0].inline_data.data

                # Convert bytes → PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")

            # PIL → CHW tensor in [0,1]
            image_tensor = transforms.ToTensor()(image)  # (3, H, W)

            # CHW → HWC
            image_tensor = image_tensor.permute(1, 2, 0)  # (H, W, 3)

            # HWC → BHWC (batch dimension)
            image_tensor = image_tensor.unsqueeze(0)  # (1, H, W, 3)

            # Ensure float32
            image_tensor = image_tensor.float()

            return (image_tensor, google_genai_client,)
        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {e}")

class GoogleI2I:
    CATEGORY = "CreatureNodes/Google"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "google_genai_client": ("ANY",),
                "images": ("STRING",),
                "model": (["Gemini 3 Pro Image Preview"],),
                "aspect_ratio": (["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"],),
                "image_size": (["1K", "2K", "4K"],),
                "prompt": ("STRING", {
                    "multiline": True
                }),
            }
        }
    RETURN_TYPES = ("IMAGE", "ANY",)
    RETURN_NAMES = ("generated_image", "google_genai_client",)
    FUNCTION = "google_i2i"
    
    def google_i2i(self, google_genai_client, images, model, aspect_ratio, image_size, prompt):
        try:
            
            images = [Image.open(image) for image in images]
            
            # Use generate_content for other models
            response = google_genai_client.models.generate_content(
                model=modelEnum[model],
                contents=[prompt, *images],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=image_size
                    )
                )
            )

            # Extract the inline_data
            image_parts = [part for part in response.parts if part.inline_data]
            image_bytes = image_parts[0].inline_data.data

            # Convert bytes → PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")

            # PIL → CHW tensor in [0,1]
            image_tensor = transforms.ToTensor()(image)  # (3, H, W)

            # CHW → HWC
            image_tensor = image_tensor.permute(1, 2, 0)  # (H, W, 3)

            # HWC → BHWC (batch dimension)
            image_tensor = image_tensor.unsqueeze(0)  # (1, H, W, 3)

            # Ensure float32
            image_tensor = image_tensor.float()

            return (image_tensor,google_genai_client,)
        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {e}")

"""
Model switch nodes live in switches.py (no heavy imports, so they can be unit tested).
"""
from .switches import T2IModelSwitch, I2VModelSwitch, I2IModelSwitch, ImageEditingModelSwitch  # noqa: E402,F401
from .longcat import LONGCAT_NODES  # noqa: E402


class LMST2T:
    CATEGORY = "CreatureNodes/LM Studio"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING",),
                "system_prompt": ("STRING", {
                    "multiline": True
                }),
                "model": (["IBM Granite 4 Micro", "Gemma 3 1B", "Qwen 3.8 27B", "Qwen 3.5 9B"],),
                "port": ("INT", {
                    "default": 1234
                })
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "t2t_lms"
    
    def t2t_lms(self, prompt, system_prompt, model, port):
        
        modelEnum = {
            "IBM Granite 4 Micro": "ibm/granite-4-micro",
            "Gemma 3 1B": "google/gemma-3-1b",
            "Qwen 3.8 27B": "qwen/qwen3.8-27b",
            "Qwen 3.5 9B": "qwen/qwen3.5-9b"
        }
        
        response = requests.post(f'http://localhost:{port}/api/v1/chat', json={
            "model": modelEnum[model],
            "system_prompt": system_prompt,
            "input": prompt
        })
        
        response_data = response.json()
        
        if 'error' in response_data and response_data['error']:
            raise RuntimeError(response_data['error']['message'])
        
        return (response_data['output'][-1]['content'],)

class LMSI2T:
    CATEGORY = "CreatureNodes/LM Studio"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING",),
                "image": ("IMAGE",),
                "system_prompt": ("STRING", {
                    "multiline": True
                }),
                "model": (["GLM 4.6V Flash"],),
                "port": ("INT", {
                    "default": 1234
                })
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "i2t_lms"
    
    def i2t_lms(self, prompt, image, system_prompt, model, port):
        
        modelEnum = {
            "GLM 4.6V Flash": "zai-org/glm-4.6v-flash"
        }
        
        # Convert image tensor to PIL
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Save to buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        # Encode to base64
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{img_str}"
        
        response = requests.post(f'http://localhost:{port}/api/v1/chat', json={
            "model": modelEnum[model],
            "system_prompt": system_prompt,
            "input": [
                {
                    "type": "text",
                    "content": prompt
                },
                {
                    "type": "image",
                    "data_url": data_url
                }
            ]
        })
        
        response_data = response.json()
        
        if 'error' in response_data and response_data['error']:
            raise RuntimeError(response_data['error']['message'])
        
        return (response_data['output'][-1]['content'],)

NODE_CLASS_MAPPINGS = {
    "Google T2T": GoogleT2T,
    "Google T2I": GoogleT2I,
    "Google I2I": GoogleI2I,
    "Google GenAI Client": GoogleGenAIClient,
    "T2I Model Switch": T2IModelSwitch,
    "I2V Model Switch": I2VModelSwitch,
    "I2I Model Switch": I2IModelSwitch,
    "Image Editing Model Switch": ImageEditingModelSwitch,
    "LM Studio T2T": LMST2T,
    "LM Studio I2T": LMSI2T,
    **LONGCAT_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Google T2T": "Google T2T",
    "Google T2I": "Google T2I",
    "Google I2I": "Google I2I",
    "Google GenAI Client": "Google GenAI Client",
    "T2I Model Switch": "T2I Model Switch",
    "I2V Model Switch": "I2V Model Switch",
    "I2I Model Switch": "I2I Model Switch",
    "Image Editing Model Switch": "Image Editing Model Switch",
    "LM Studio T2T": "LM Studio T2T",
    "LM Studio I2T": "LM Studio I2T",
    **{name: f"{name} (CreatureNodes)" for name in LONGCAT_NODES},
}