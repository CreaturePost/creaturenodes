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
Model Switch node for image generation
"""

class T2IModelSwitch:
    CATEGORY = "CreatureNodes/Switches"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "selected_model": (["SD3.5", "SDXL", "Z IMAGE", "Flux2 Klein"],)
            },
            "optional": {
                "sd35": ("IMAGE", {"lazy": True}),
                "sdxl": ("IMAGE", {"lazy": True}),
                "z_image": ("IMAGE", {"lazy": True}),
                "flux2_klein": ("IMAGE", {"lazy": True}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "t2i_model_switch"
    
    def check_lazy_status(
        self,
        selected_model,
        sd35=None,
        sdxl=None, 
        z_image=None,
        flux2_klein=None
    ):
        needed = []
        if selected_model == "SD3.5":
            if sd35 is None: needed.append("sd35")
        elif selected_model == "SDXL":
            if sdxl is None: needed.append("sdxl")
        elif selected_model == "Z IMAGE":
            if z_image is None: needed.append("z_image")
        elif selected_model == "Flux2 Klein":
            if z_image is None: needed.append("flux2_klein")
        
        return needed
    
    def t2i_model_switch(self, selected_model, sd35=None, sdxl=None, z_image=None, flux2_klein=None):
        
        modelEnum = {
            'SD3.5': sd35,
            'SDXL': sdxl,
            'Z IMAGE': z_image,
            'Flux2 Klein': flux2_klein
        }
        
        return (modelEnum[selected_model], )

class I2VModelSwitch:
    CATEGORY = "CreatureNodes/Switches"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "selected_model": (["LTX 2.0", "MiniMax H3", "MiniMax H3 Ref"],)
            },
            "optional": {
                "ltx_20_image": ("IMAGE", {"lazy": True}),
                "ltx_20_audio": ("AUDIO", {"lazy": True}),
                "mm_h3_image": ("IMAGE", {"lazy": True}),
                "mm_h3_audio": ("AUDIO", {"lazy": True}),
                "mm_h3_ref_image": ("IMAGE", {"lazy": True}),
                "mm_h3_ref_audio": ("AUDIO", {"lazy": True}),
            }
        }
    RETURN_TYPES = ("IMAGE", "AUDIO",)
    RETURN_NAMES = ("image", "audio",)
    FUNCTION = "i2v_model_switch"
    
    def check_lazy_status(
        self,
        selected_model,
        ltx_20_image=None,
        ltx_20_audio=None,
        mm_h3_image=None,
        mm_h3_audio=None,
        mm_h3_ref_image=None,
        mm_h3_ref_audio=None
    ):
        needed = []
        if selected_model == "LTX 2.0":
            if ltx_20_image is None: needed.append("ltx_20_image")
            if ltx_20_audio is None: needed.append("ltx_20_audio")
        elif selected_model == "MiniMax H3":
            if mm_h3_image is None: needed.append("mm_h3_image")
            if mm_h3_audio is None: needed.append("mm_h3_audio")
        elif selected_model == "MiniMax H3 Ref":
            if mm_h3_ref_image is None: needed.append("mm_h3_ref_image")
            if mm_h3_ref_audio is None: needed.append("mm_h3_ref_audio")
        
        return needed
    
    def i2v_model_switch(self, selected_model, ltx_20_image=None, ltx_20_audio=None, mm_h3_image=None, mm_h3_audio=None, mm_h3_ref_image=None, mm_h3_ref_audio=None):
        
        modelEnum = {
            'LTX 2.0': (ltx_20_image, ltx_20_audio,),
            'MiniMax H3': (mm_h3_image, mm_h3_audio,),
            'MiniMax H3 Ref': (mm_h3_ref_image, mm_h3_ref_audio,),
        }
        
        return modelEnum[selected_model]

class I2IModelSwitch:
    CATEGORY = "CreatureNodes/Switches"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "selected_model": (["Flux1 Dev Uso", "Flux1 Kontext"],)
            },
            "optional": {
                "flux1_dev_uso": ("IMAGE", {"lazy": True}),
                "flux1_kontext": ("IMAGE", {"lazy": True}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "i2i_model_switch"
    
    def check_lazy_status(
        self,
        selected_model,
        flux1_dev_uso=None,
        flux1_kontext=None
    ):
        needed = []
        if selected_model == "Flux1 Dev Uso":
            if flux1_dev_uso is None: needed.append("flux1_dev_uso")
        elif selected_model == "Flux1 Kontext":
            if flux1_kontext is None: needed.append("flux1_kontext")
        
        return needed
    
    def i2i_model_switch(self, selected_model, flux1_dev_uso=None, flux1_kontext=None):
        
        modelEnum = {
            "Flux1 Dev Uso": flux1_dev_uso,
            "Flux1 Kontext": flux1_kontext
        }
        
        return (modelEnum[selected_model],)

class ImageEditingModelSwitch:
    CATEGORY = "CreatureNodes/Switches"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "selected_model": (["Lucy Edit V1.1 Dev", "Seedance 2.5"],)
            },
            "optional": {
                "lucy_edit_v11_dev": ("IMAGE", {"lazy": True}),
                "seedance_25": ("IMAGE", {"lazy": True})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_editing_model_switch"
    
    def check_lazy_status(
        self,
        selected_model,
        lucy_edit_v11_dev=None,
        seedance_25=None
    ):
        needed = []
        if selected_model == "Lucy Edit V1.1 Dev":
            if lucy_edit_v11_dev is None: needed.append("lucy_edit_v11_dev")
        elif selected_model == "Seedance 2.5":
            if seedance_25 is None: needed.append("seedance_25")
        
        return needed
    
    def image_editing_model_switch(self, selected_model, lucy_edit_v11_dev=None, seedance_25=None):
        
        modelEnum = {
            "Lucy Edit V1.1 Dev": lucy_edit_v11_dev,
            "Seedance 2.5": seedance_25
        }
        
        return (modelEnum[selected_model],)
    
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
    "LM Studio I2T": LMSI2T
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
    "LM Studio I2T": "LM Studio I2T"
}